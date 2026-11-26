"""기본 측정 (0단계). 정밀 측정(P3 전체)은 1단계에서 이 파일을 넓힌다.

- 수평 각도: 눈 바깥 꼭짓점 (가이드 §4)
- 얼굴 중심선: 두 눈 안쪽 끝 중간 (가이드 §15-2)
- 정수리: 중심선 주변 띠에서 사람 영역(머리카락 포함)이 시작되는 가장 위 행
- 사람 영역 경계 다듬기: 가이드 필터 (He et al., 2010 논문의 공개 식)
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def tilt_degrees(points: dict[str, tuple[float, float]]) -> float:
    """사진 왼쪽 → 오른쪽 눈 바깥 꼭짓점 선의 기울기 (도). +면 오른쪽이 아래."""
    (x1, y1), (x2, y2) = points["왼눈_바깥꼭짓점"], points["오른눈_바깥꼭짓점"]
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def center_x(points: dict[str, tuple[float, float]]) -> float:
    return (points["왼눈_안쪽꼭짓점"][0] + points["오른눈_안쪽꼭짓점"][0]) / 2


def face_width(points: dict[str, tuple[float, float]]) -> float:
    if "왼얼굴_옆" in points and "오른얼굴_옆" in points:
        return abs(points["오른얼굴_옆"][0] - points["왼얼굴_옆"][0])
    return 2.2 * abs(points["오른눈_바깥꼭짓점"][0] - points["왼눈_바깥꼭짓점"][0]) / 1.4


def crown_y(person_mask: np.ndarray, cx: float, band_half: float, thr: float = 0.5,
            min_fill: float = 0.3) -> float | None:
    """중심선 ± band_half 띠에서 사람 영역이 min_fill 이상 차는 첫 행."""
    h, w = person_mask.shape[:2]
    x0, x1 = max(0, int(cx - band_half)), min(w, int(cx + band_half) + 1)
    if x1 <= x0:
        return None
    band = person_mask[:, x0:x1]
    fill = (band > thr).mean(axis=1)
    rows = np.flatnonzero(fill >= min_fill)
    if rows.size == 0:
        return None
    return float(rows[0])


def _box(x: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """회색조 가이드 필터. guide, src: float32 0~1."""
    mean_i, mean_p = _box(guide, r), _box(src, r)
    cov = _box(guide * src, r) - mean_i * mean_p
    var = _box(guide * guide, r) - mean_i * mean_i
    a = cov / (var + eps)
    b = mean_p - a * mean_i
    return _box(a, r) * guide + _box(b, r)


def clean_confidence(m: np.ndarray, lo: float = 0.15, hi: float = 0.85) -> np.ndarray:
    """분할 모델은 배경에도 0.1 안팎의 값을 남긴다. lo 이하는 0, hi 이상은 1로 펴 준다."""
    return np.clip((m.astype(np.float32) - lo) / (hi - lo), 0, 1)


def refine_mask(rgb: np.ndarray, coarse: np.ndarray, work_side: int = 2000) -> np.ndarray:
    """거친 사람 영역을 사진 경계(머리카락 끝)에 맞게 다듬는다. 결과는 원본 크기 float32 0~1."""
    coarse = clean_confidence(coarse)
    h, w = rgb.shape[:2]
    s = min(1.0, work_side / max(h, w))
    size = (max(1, round(w * s)), max(1, round(h * s)))
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA) if s < 1 else rgb
    m = cv2.resize(coarse, size, interpolation=cv2.INTER_AREA) if s < 1 else coarse
    g = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    r = max(2, round(max(size) / 250))
    out = guided_filter(g, m.astype(np.float32), r, 1e-3)
    out = np.clip(out, 0, 1)
    if s < 1:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out.astype(np.float32)
