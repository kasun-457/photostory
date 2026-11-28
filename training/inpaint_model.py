"""3-5 부분 생성형 모델: 좁은 구멍 채우기 (학습설계 2 · 3-5).

LaMa 논문의 생각(빠른 푸리에 합성곱 = 사진 전체를 한 번에 보는 층)을 따라 처음부터 짠 작은 U-Net.
남의 가중치는 쓰지 않는다 (학습설계 0-4).
  입력 : 구멍 뚫린 사진 (3) + 구멍 (1)
  출력 : 채운 사진 (3, 0~1). 쓸 때는 구멍 밖은 입력 그대로 (compose)
판별기: 작은 PatchGAN (피부 결·머리카락 결이 뭉개지지 않게).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(cin, cout, k=3, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.GroupNorm(8, cout), nn.SiLU())


class FourierUnit(nn.Module):
    """주파수 공간 1×1 합성곱 = 사진 전체를 보는 층."""

    def __init__(self, c: int):
        super().__init__()
        self.conv = nn.Conv2d(2 * c, 2 * c, 1, bias=False)
        self.norm = nn.GroupNorm(8, 2 * c)

    def forward(self, x):
        h, w = x.shape[-2:]
        dt = x.dtype
        f = torch.fft.rfft2(x.float(), norm="ortho")
        y = torch.cat([f.real, f.imag], 1)
        with torch.autocast(x.device.type, enabled=False):
            y = F.silu(self.norm(self.conv(y.float())))
        re, im = y.chunk(2, 1)
        return torch.fft.irfft2(torch.complex(re, im), s=(h, w), norm="ortho").to(dt)


class FFCBlock(nn.Module):
    """채널 절반은 가까운 곳(3×3), 절반은 사진 전체(푸리에). 잔차 연결."""

    def __init__(self, c: int):
        super().__init__()
        self.cl = c // 2
        self.local = nn.Sequential(conv(c, self.cl), nn.Conv2d(self.cl, self.cl, 3, 1, 1))
        self.glob_in = nn.Conv2d(c, c - self.cl, 1)
        self.fu = FourierUnit(c - self.cl)
        self.glob_out = nn.Conv2d(c - self.cl, c - self.cl, 1)
        self.norm = nn.GroupNorm(8, c)

    def forward(self, x):
        g = self.glob_in(x)
        g = self.glob_out(g + self.fu(g))
        return x + F.silu(self.norm(torch.cat([self.local(x), g], 1)))


class InpaintNet(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256), blocks: int = 6):
        super().__init__()
        w = widths
        self.inp = nn.Sequential(nn.Conv2d(4, w[0], 5, 1, 2), nn.SiLU())
        self.down = nn.ModuleList([conv(w[i], w[i + 1], 4, 2) for i in range(len(w) - 1)])
        self.enc = nn.ModuleList([conv(c, c) for c in w[:-1]])
        self.mid = nn.Sequential(*[FFCBlock(w[-1]) for _ in range(blocks)])
        self.up = nn.ModuleList([conv(w[i + 1], w[i]) for i in reversed(range(len(w) - 1))])
        self.dec = nn.ModuleList([conv(2 * w[i], w[i]) for i in reversed(range(len(w) - 1))])
        self.out = nn.Conv2d(w[0], 3, 3, 1, 1)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.inp(torch.cat([img * (1 - mask), mask], 1))
        skips = []
        for e, d in zip(self.enc, self.down):
            x = e(x)
            skips.append(x)
            x = d(x)
        x = self.mid(x)
        for u, dc in zip(self.up, self.dec):
            s = skips.pop()
            x = u(F.interpolate(x, size=s.shape[-2:], mode="nearest"))
            x = dc(torch.cat([x, s], 1))
        return torch.sigmoid(self.out(x).float())


def compose(img: torch.Tensor, mask: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return img * (1 - mask) + pred * mask


class PatchDisc(nn.Module):
    def __init__(self, c: int = 64):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv2d(4, c, 4, 2, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(c, 2 * c, 4, 2, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(2 * c, 4 * c, 4, 2, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(4 * c, 4 * c, 3, 1, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(4 * c, 1, 3, 1, 1)))

    def forward(self, img, mask):
        return self.net(torch.cat([img, mask], 1).float())
