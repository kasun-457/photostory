"""정밀 측정 P3: 코 중심·입 중심 (가이드 §9-2 '코밑_x_중심선차', '입중심_x_중심선차').

MediaPipe 얼굴 점은 평균 얼굴 모양 쪽으로 끌려서, 코·입이 옆으로 치우친 만큼의 절반 정도만 잰다
(코·입을 12px 옮긴 사진에서 5.6~7px). 그래서 사진에서 다시 잰다.
- 코 중심: 코 아래 영역에서 가장 어두운 두 덩어리(콧구멍)의 가운데.
- 입 중심: 입술 색(주변 피부보다 붉음)으로 잡은 입술 영역의 좌우 끝 가운데.
찾지 못하거나 애매하면 None → 얼굴 점을 그대로 쓴다 (보수적).
기준값은 공개 사진으로 잡은 값이다 (확실하지 않음).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import regions as R


def _crop(img, poly, grow):
    h, w = img.shape[:2]
    x0, y0 = np.floor(poly.min(axis=0) - grow).astype(int)
    x1, y1 = np.ceil(poly.max(axis=0) + grow).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    return img[y0:y1, x0:x1], (x0, y0)


def nose_center(rgb: np.ndarray, regions, points) -> float | None:
    """두 콧구멍 무게중심의 가운데 x.

    조명이 한쪽에서 오면 한쪽 콧구멍이 훨씬 밝다 → 좌우 반쪽마다 따로 가장 어두운 덩어리를 찾는다.
    콧볼 바깥 주름 그림자를 잡지 않도록 바깥 15%는 뺀다.
    """
    hull = regions.get("코아래")
    if hull is None or "코밑" not in points:
        return None
    crop, off = _crop(rgb, hull, 2)
    if crop.size < 50:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = R.poly_mask(g.shape, hull, hull=True, offset=off) > 0
    xs_in = np.nonzero(m.any(axis=0))[0]
    if xs_in.size < 10:
        return None
    lo, hi = xs_in[0], xs_in[-1]
    # 좌우를 가르는 곳: 두 콧볼의 가운데 (코밑 점보다 좌우 대칭이 안정적)
    if "왼콧볼" in points and "오른콧볼" in points:
        mid = (points["왼콧볼"][0] + points["오른콧볼"][0]) / 2 - off[0]
    else:
        mid = points["코밑"][0] - off[0]
    edge = 0.15 * (hi - lo)
    cents = []
    for x0, x1 in ((lo + edge, mid), (mid, hi - edge)):
        half = m.copy()
        half[:, :max(0, int(x0))] = False
        half[:, int(x1):] = False
        vals = g[half]
        if vals.size < 15:
            return None
        thr = np.percentile(vals, 15)
        if float(np.median(vals)) - float(thr) < 12:        # 이쪽 콧구멍이 주변과 구분되지 않음
            return None
        dark = ((g <= thr) & half).astype(np.uint8)
        n, _, st, cen = cv2.connectedComponentsWithStats(dark, connectivity=8)
        if n < 2:
            return None
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        cents.append((float(cen[i][0]), int(st[i, cv2.CC_STAT_AREA])))
    (x1, a1), (x2, a2) = cents
    if min(a1, a2) < 0.2 * max(a1, a2):                     # 크기가 너무 다르면 콧구멍 한 쌍이 아님
        return None
    return float((x1 + x2) / 2 + off[0])


def mouth_center(rgb: np.ndarray, regions, points) -> float | None:
    """입술 영역(붉은 기) 좌우 끝의 가운데 x."""
    lips = regions.get("입술")
    if lips is None:
        return None
    mw = float(np.ptp(lips[:, 0]))
    crop, off = _crop(rgb, lips, 0.35 * mw)
    if crop.size < 50:
        return None
    a = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)[..., 1].astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), max(0.6, 0.01 * mw))
    inside = R.poly_mask(a.shape, lips, hull=True, offset=off)
    ring = R.dilate(inside, 0.25 * mw) & ~R.dilate(inside, 0.08 * mw)
    if not ring.any() or not inside.any():
        return None
    skin = float(np.median(a[ring > 0]))
    lip = float(np.percentile(a[inside > 0], 75))
    if lip - skin < 3:                                       # 입술이 피부와 색으로 구분되지 않음
        return None
    mask = (a > skin + 0.5 * (lip - skin)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None
    # 얼굴 점 입술 영역과 가장 많이 겹치는 덩어리들 (윗입술·아랫입술이 갈라져도 함께)
    keep = [i for i in range(1, n) if (inside[lab == i] > 0).mean() > 0.5
            and st[i, cv2.CC_STAT_AREA] > 0.02 * inside.sum() / 255]
    if not keep:
        return None
    xs = np.concatenate([np.nonzero(lab == i)[1] for i in keep])
    l, r = np.percentile(xs, 0.5), np.percentile(xs, 99.5)
    width = r - l
    if not (0.7 * mw <= width <= 1.4 * mw):                 # 입 폭이 얼굴 점과 너무 다르면 믿지 않음
        return None
    return float((l + r) / 2 + off[0])


def measure(rgb: np.ndarray, regions, points) -> dict[str, tuple[float, float]]:
    """prepare용. 찾은 것만 표준 점 '코_중심'·'입_중심'으로 돌려준다 (y는 얼굴 점)."""
    out = {}
    nx = nose_center(rgb, regions, points)
    if nx is not None:
        out["코_중심"] = (nx, points["코밑"][1])
    mx = mouth_center(rgb, regions, points)
    if mx is not None and "왼입꼬리" in points:
        y = (points["왼입꼬리"][1] + points["오른입꼬리"][1]) / 2
        out["입_중심"] = (mx, y)
    return out
