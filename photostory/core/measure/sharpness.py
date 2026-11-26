"""눈·코·입이 흐린지 재기 (가이드 §7: 흐릿해 보일 때만 선명하게, 애매하면 하지 않음).

얼굴 폭을 300px(전송본에 찍히는 크기와 비슷)으로 맞춘 뒤, 눈·코·입 둘레에서
'살짝(σ=1) 흐렸을 때 사라지는 경계 세기 / 원래 경계 세기'를 잰다. 또렷한 사진일수록 크다.
해상도가 달라도 같은 사진이면 거의 같은 값이 나온다 (512px 공개 사진을 1~3배 키워도 0.09 안팎).

기준값 SOFT는 공개 사진 하나를 흐려 가며 정한 값이다 (흐림 없음 0.091, σ1 0.068, σ2 0.041).
실제 사진관 원본으로 다시 맞춰야 한다 (확실하지 않음).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import regions as R

NORM_FACE_W = 300.0
SOFT = 0.075          # 이보다 작으면 흐린 것으로 보고 선명하게 한다
PARTS = ("왼눈", "오른눈", "입술", "코아래")


def _grad(x: np.ndarray) -> np.ndarray:
    return np.hypot(cv2.Sobel(x, cv2.CV_32F, 1, 0), cv2.Sobel(x, cv2.CV_32F, 0, 1))


def feature_sharpness(rgb: np.ndarray, points, regions) -> float | None:
    if not all(k in regions for k in PARTS) or "얼굴윤곽" not in regions:
        return None
    fw = R.face_w(points)
    s = NORM_FACE_W / max(fw, 1.0)
    x0, y0, x1, y1 = R.face_box(regions, rgb.shape, 0.05 * fw)
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    off = np.array([x0, y0], np.float32)
    m = np.zeros(g.shape, np.uint8)
    for k in PARTS:
        m |= R.poly_mask(g.shape, (regions[k] - off) * s, hull=True)
    m = R.dilate(m, 3) > 0
    if not m.any():
        return None
    G = _grad(g)
    lost = G - _grad(cv2.GaussianBlur(g, (0, 0), 1.0))
    return float(lost[m].mean() / max(1e-6, float(G[m].mean())))
