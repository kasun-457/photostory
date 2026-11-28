"""3-5 부분 생성형 학습 데이터 (학습설계 2 · 3-5).

정답 지도의 '메움'(fill) 자리 = 유동화·톤·피부로 설명되지 않아 성현님이 새로 칠한 곳 (잔머리, 핀, 콧볼 옆 그림자, 큰 잡티).
앱에서는 보정 지도를 적용한 사진에서 메움 자리를 지우고 이 모델이 채운다. 둘레는 이미 수정본에 가까우므로
학습에서는 **수정본(r)에서 구멍을 뚫고 수정본으로 채우게** 한다. 원본의 구멍 속은 보여 주지 않는다 (핀·잔머리를 따라 그리지 않게).

한 개 = 조각 해상도(1408×1728)에서 patch×patch 하나
  img  (3,P,P)  수정본 0~1 (정답, 입력은 img·(1−mask))
  mask (1,P,P)  구멍 1
  loss (1,P,P)  손실을 볼 곳 (조각 안 유효 영역)
구멍: 확률 1−synth 로 실제 메움 덩어리 둘레 (조금 넓힘), synth 로 사람 영역 안 임의 얼룩·획 (메움이 드문 곳도 배우게).
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .crop_pairs import load_npz

FILL_GROW = 3          # 메움 덩어리를 이만큼(px) 넓혀 구멍으로 (경계 이음매가 정답 쪽에 남지 않게)


def list_items(work: Path, split: str | None = None) -> list[str]:
    from .dataset import list_items as li
    return li(work, split)


def synth_mask(rng: np.random.Generator, p: int) -> np.ndarray:
    """임의 얼룩·획 (잡티·잔머리·핀 크기). 0/1 uint8."""
    m = np.zeros((p, p), np.uint8)
    for _ in range(rng.integers(1, 5)):
        if rng.random() < 0.5:                      # 얼룩
            c = rng.integers(p // 4, 3 * p // 4, 2)
            ax = rng.integers(4, p // 6, 2)
            cv2.ellipse(m, (int(c[0]), int(c[1])), (int(ax[0]), int(ax[1])), float(rng.uniform(0, 180)), 0, 360, 1, -1)
        else:                                       # 획 (잔머리·핀)
            pts = [rng.integers(p // 6, 5 * p // 6, 2)]
            for _ in range(rng.integers(2, 6)):
                step = rng.normal(0, p / 6, 2)
                pts.append(np.clip(pts[-1] + step, 0, p - 1).astype(int))
            w = int(rng.integers(2, 14))
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(m, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 1, w)
    return m


class InpaintDataset(Dataset):
    def __init__(self, work: Path, keys: list[str], patch: int = 256, augment: bool = False,
                 synth: float = 0.3, fixed_seed: int | None = None):
        """fixed_seed가 있으면 i번째 항목은 언제나 같은 조각·구멍 (검증용)."""
        self.work, self.keys, self.p = Path(work), keys, patch
        self.augment, self.synth, self.fixed_seed = augment, synth, fixed_seed

    def __len__(self):
        return len(self.keys)

    def _rng(self, i: int) -> np.random.Generator:
        if self.fixed_seed is not None:
            return np.random.default_rng(self.fixed_seed * 1_000_003 + i)
        return np.random.default_rng()

    def sample(self, key: str, rng: np.random.Generator) -> dict:
        p = self.p
        crops, tg = self.work / "crops", self.work / "targets"
        r = cv2.cvtColor(cv2.imread(str(crops / f"{key}_r.jpg")), cv2.COLOR_BGR2RGB)
        h, w = r.shape[:2]
        with np.load(tg / f"{key}.npz") as t:
            fill = t["fill"] > 127
        meta = load_npz(crops / f"{key}.npz")
        person = meta["masks"]["person"]
        if person.shape != (h, w):
            person = cv2.resize(person, (w, h), interpolation=cv2.INTER_NEAREST)
        valid = meta["valid"]
        if valid.shape != (h, w):
            valid = cv2.resize(valid, (w, h), interpolation=cv2.INTER_NEAREST)

        use_fill = fill.any() and rng.random() >= self.synth
        pool = np.flatnonzero(fill) if use_fill else np.flatnonzero(person > 128)
        if pool.size == 0:
            pool = np.arange(h * w)
        cy, cx = divmod(int(pool[rng.integers(pool.size)]), w)
        y0 = int(np.clip(cy - p // 2 + rng.integers(-p // 4, p // 4 + 1), 0, h - p))
        x0 = int(np.clip(cx - p // 2 + rng.integers(-p // 4, p // 4 + 1), 0, w - p))
        sl = np.s_[y0:y0 + p, x0:x0 + p]
        img = r[sl].astype(np.float32) / 255
        if use_fill:
            m = cv2.dilate(fill[sl].astype(np.uint8), np.ones((2 * FILL_GROW + 1,) * 2, np.uint8))
        else:
            m = synth_mask(rng, p) * (person[sl] > 128)
            if not m.any():
                m = synth_mask(rng, p)
        lw = (valid[sl] > 128).astype(np.float32)
        m = m.astype(np.float32) * lw
        if self.augment and rng.random() < 0.5:
            img, m, lw = img[:, ::-1], m[:, ::-1], lw[:, ::-1]
        return dict(img=torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                    mask=torch.from_numpy(np.ascontiguousarray(m))[None],
                    loss=torch.from_numpy(np.ascontiguousarray(lw))[None])

    def __getitem__(self, i):
        return self.sample(self.keys[i], self._rng(i))


def telea_baseline(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """비교 기준: OpenCV 고전 채우기 (img 0~1 H×W×3, mask H×W 0/1) → 0~1."""
    u8 = (img * 255 + 0.5).astype(np.uint8)
    out = cv2.inpaint(u8, (mask > 0.5).astype(np.uint8), 5, cv2.INPAINT_TELEA)
    return out.astype(np.float32) / 255


def keys_with_fill(work: Path, keys: list[str]) -> list[str]:
    with open(Path(work) / "targets.csv", newline="", encoding="utf-8-sig") as fh:
        ff = {r["key"]: float(r["fill_frac"] or 0) for r in csv.DictReader(fh) if r["status"] == "ok"}
    return [k for k in keys if ff.get(k, 0) > 0]
