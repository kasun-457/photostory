"""3-4 보정 지도 모델 (학습설계 3).

입력: 원본(1/2 해상도) + 좌우 뒤집은 원본 + 얼굴 부위·영역 12채널 + 좌표 2채널
  - 뒤집은 원본: 조각은 두 눈 가운데가 세로 중심이라, 뒤집으면 반대쪽 얼굴이 같은 자리에 온다.
    대칭 보정(좁은 쪽에 맞추기)을 배우는 데 필요한 '반대쪽'을 바로 보게 한다.
  - 가운데(1/16)에 주의(attention) 층 2개: 얼굴 전체를 한 번에 보고 어디를 얼마나 고칠지 정함.
출력 (1/2 해상도):
  flow (2) 유동화 px · alpha (1) 피부 비율 0~1 · gain (3) 톤 배율 log (1/4로 줄여 씀) · fill (1) 메움 로짓

남의 가중치를 쓰지 않고 처음부터 학습한다 (학습설계 0-4).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import N_AUX

GAIN_MAX = 0.7           # log 배율 한계 (≈ ×0.5 ~ ×2)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class CorrectionMapNet(nn.Module):
    def __init__(self, widths=(16, 48, 96, 192, 256), attn_layers: int = 2, heads: int = 8,
                 checkpoint: bool = True):
        super().__init__()
        self.checkpoint = checkpoint          # 메모리 절약 (RTX 3060 12GB): 중간값을 버렸다가 역전파 때 다시 계산
        cin = 3 + 3 + N_AUX + 2
        self.stem = nn.Conv2d(cin, widths[0], 3, padding=1)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i, w in enumerate(widths):
            self.enc.append(nn.Sequential(ResBlock(widths[max(i - 1, 0)] if i else widths[0], w), ResBlock(w, w)))
            if i < len(widths) - 1:
                self.down.append(nn.Conv2d(w, w, 3, stride=2, padding=1))
        d = widths[-1]
        self.attn = nn.ModuleList(nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.0, batch_first=True,
                                                             norm_first=True) for _ in range(attn_layers))
        self.pos = nn.Conv2d(d, d, 3, padding=1, groups=d)            # 위치 정보 (깊이별 합성곱)
        self.dec = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.dec.append(nn.Sequential(ResBlock(widths[i] + widths[i - 1], widths[i - 1]),
                                          ResBlock(widths[i - 1], widths[i - 1])))
        self.out_norm = nn.GroupNorm(8, widths[0])
        self.head = nn.Conv2d(widths[0], 2 + 1 + 3 + 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[2] = -2.0          # 피부 비율은 0 근처에서 시작
            self.head.bias[6] = -4.0          # 메움도 거의 없음에서 시작

    def _run(self, blk, x):
        if self.checkpoint and self.training and x.requires_grad:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(blk, x, use_reentrant=False)
        return blk(x)

    def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, h, w = rgb.shape
        ys = torch.linspace(-1, 1, h, device=rgb.device, dtype=rgb.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1, 1, w, device=rgb.device, dtype=rgb.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        x = torch.cat([rgb * 2 - 1, torch.flip(rgb, dims=[3]) * 2 - 1, aux, xs, ys], 1)
        x = self.stem(x)
        skips = []
        for i, blk in enumerate(self.enc):
            x = self._run(blk, x)
            if i < len(self.down):
                skips.append(x)
                x = self.down[i](x)
        x = x + self.pos(x)
        bb, c, hh, ww = x.shape
        tok = x.flatten(2).transpose(1, 2)
        for layer in self.attn:
            tok = layer(tok)
        x = tok.transpose(1, 2).reshape(bb, c, hh, ww)
        for blk in self.dec:
            s = skips.pop()
            x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = self._run(blk, torch.cat([x, s], 1))
        y = self.head(F.silu(self.out_norm(x))).float()
        return dict(flow=y[:, 0:2] * 4.0,                       # px. 큰 유동화(수십 px)도 낼 수 있게 배율
                    alpha=torch.sigmoid(y[:, 2:3]),
                    gain=torch.tanh(y[:, 3:6]) * GAIN_MAX,      # 1/2 해상도 (손실에서 1/4로 줄여 비교)
                    fill=y[:, 6:7])


def apply_maps(rgb: torch.Tensor, s1: torch.Tensor, flow: torch.Tensor, affine: torch.Tensor | None,
               gain: torch.Tensor, alpha: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """지도를 원본에 적용 (학습 손실과 앱이 같은 식을 쓴다).

    수정본 화소 x = 원본의 A·x + flow(x) → × exp(gain) → 피부: + alpha·(S1 − t).
    strength는 유동화·톤·피부에 함께 곱한다 (배치 A에는 곱하지 않음).
    """
    b, _, h, w = rgb.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=rgb.device, dtype=torch.float32),
                            torch.arange(w, device=rgb.device, dtype=torch.float32), indexing="ij")
    if affine is None:
        px, py = xs.expand(b, h, w), ys.expand(b, h, w)
    else:
        a = affine.float()
        px = a[:, 0, 0, None, None] * xs + a[:, 0, 1, None, None] * ys + a[:, 0, 2, None, None]
        py = a[:, 1, 0, None, None] * xs + a[:, 1, 1, None, None] * ys + a[:, 1, 2, None, None]
    px = px + strength * flow[:, 0]
    py = py + strength * flow[:, 1]
    grid = torch.stack([px / (w - 1) * 2 - 1, py / (h - 1) * 2 - 1], -1)
    both = F.grid_sample(torch.cat([rgb, s1], 1).float(), grid, mode="bilinear",
                         padding_mode="border", align_corners=True)
    ow, sw = both[:, :3], both[:, 3:]
    g = gain if gain.shape[-2:] == (h, w) else F.interpolate(gain, size=(h, w), mode="bilinear", align_corners=False)
    g = torch.exp(strength * g)
    t, s = ow * g, sw * g
    return t + (strength * alpha).clamp(0, 1) * (s - t)
