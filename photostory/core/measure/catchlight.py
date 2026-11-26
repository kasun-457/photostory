"""눈 반사점 판정 (설계서 3-2의 5번, v1.52 '반대쪽 눈 따오기').

조명 반사가 눈동자를 크게 덮어 '뚫려 보이는' 눈이 한쪽에만 있을 때만 고친다.
- 홍채 원 안에서, 홍채 보통 밝기보다 훨씬 밝은 화소 덩어리 중 가장 큰 것의 넓이 비율을 잰다.
- 한쪽이 BAD 이상이고 다른 쪽이 CLEAN 이하이며 차이가 RATIO배 이상이면 그 눈을 고친다.
- 양쪽이 다 크거나(같은 조명) 애매하면 고치지 않는다 (보수적 기본값).
기준값은 합성 눈으로 잡은 값이라 실제 사진관 원본으로 다시 맞춰야 한다 (확실하지 않음).
"""
from __future__ import annotations

import cv2
import numpy as np

BAD = 0.10       # 홍채 넓이의 10% 이상을 덮는 밝은 덩어리
CLEAN = 0.06
RATIO = 2.5
EYES = ("왼눈", "오른눈")


def iris_circle(points, eye: str) -> tuple[tuple[float, float], float] | None:
    c = points.get(f"{eye}_홍채중심")
    ring = [points[k] for k in (f"{eye}_홍채위", f"{eye}_홍채아래", f"{eye}_홍채안쪽", f"{eye}_홍채바깥")
            if k in points]
    if c is None or len(ring) < 2:
        return None
    r = float(np.mean([np.hypot(x - c[0], y - c[1]) for x, y in ring]))
    return (float(c[0]), float(c[1])), r


def iris_disk(shape, center, r, offset=(0, 0)) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.circle(m, (round(center[0] - offset[0]), round(center[1] - offset[1])), max(1, round(r)), 255, -1,
               cv2.LINE_AA)
    return m


def glare_fraction(rgb: np.ndarray, center, r: float) -> float:
    """홍채 안에서 가장 큰 '아주 밝은' 덩어리가 홍채 넓이에서 차지하는 비율."""
    h, w = rgb.shape[:2]
    x0, y0 = int(max(0, center[0] - r - 2)), int(max(0, center[1] - r - 2))
    x1, y1 = int(min(w, center[0] + r + 3)), int(min(h, center[1] + r + 3))
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    disk = iris_disk(crop.shape, center, 0.92 * r, (x0, y0)) > 0      # 가장자리(흰자 섞임) 조금 뺌
    if disk.sum() < 10:
        return 0.0
    L = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    base = float(np.percentile(L[disk], 40))                         # 홍채 보통 밝기
    bright = ((L > max(base + 0.45 * (255 - base), 170)) & disk).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n <= 1:
        return 0.0
    return float(st[1:, cv2.CC_STAT_AREA].max() / disk.sum())


def judge(rgb: np.ndarray, points) -> dict:
    """{"fix": 고칠 눈 또는 None, "glare": {눈: 비율}}"""
    glare = {}
    for e in EYES:
        ic = iris_circle(points, e)
        glare[e] = glare_fraction(rgb, *ic) if ic else 0.0
    bad = max(EYES, key=lambda e: glare[e])
    good = EYES[1 - EYES.index(bad)]
    fix = bad if (glare[bad] >= BAD and glare[good] <= CLEAN
                  and glare[bad] >= RATIO * max(glare[good], 1e-3)) else None
    return {"fix": fix, "glare": {k: round(v, 3) for k, v in glare.items()}}
