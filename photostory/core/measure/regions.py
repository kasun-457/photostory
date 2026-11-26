"""얼굴 부위 영역으로 마스크를 만드는 도구 (피부·명암 단계 공용).

가이드의 px 값(브러시 175~300px, 중간값 25px 등)은 사진관 원본(4472×6708) 기준이다.
사진마다 얼굴 크기가 다르므로 '입 높이 얼굴 폭'으로 배율을 잡는다 (face_k).
"""
from __future__ import annotations

import cv2
import numpy as np

# 가이드 예시의 입 높이 얼굴 폭: 윤지수 1196px, 이지혁 1248px (가이드 §5-1·§5-2).
# 모델 점은 실제 윤곽보다 조금 안쪽이라 실제 폭보다 작게 나온다. 샘플로 다시 맞춰야 함 (확실하지 않음).
REF_FACE_W = 1200.0


def face_w(points: dict[str, tuple[float, float]]) -> float:
    if "왼볼_입높이" in points and "오른볼_입높이" in points:
        return abs(points["오른볼_입높이"][0] - points["왼볼_입높이"][0])
    return 1.9 * abs(points["오른눈_바깥꼭짓점"][0] - points["왼눈_바깥꼭짓점"][0])


def face_k(points: dict[str, tuple[float, float]]) -> float:
    """가이드 px 값 × face_k = 지금 이미지에서의 px 값."""
    return max(1e-3, face_w(points) / REF_FACE_W)


def poly_mask(shape: tuple[int, int], poly: np.ndarray, hull: bool = False,
              offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    """닫힌 다각형 → uint8 0/255. offset은 잘라 낸 영역의 왼쪽 위 (x, y)."""
    m = np.zeros(shape[:2], np.uint8)
    p = np.asarray(poly, np.float32) - np.asarray(offset, np.float32)
    if hull:
        p = cv2.convexHull(p)
    cv2.fillPoly(m, [np.round(p).astype(np.int32).reshape(-1, 1, 2)], 255, cv2.LINE_AA)
    return m


def line_mask(shape: tuple[int, int], line: np.ndarray, thickness: float,
              offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    """꺾은선 → 굵은 띠 uint8 0/255."""
    m = np.zeros(shape[:2], np.uint8)
    p = np.asarray(line, np.float32) - np.asarray(offset, np.float32)
    cv2.polylines(m, [np.round(p).astype(np.int32).reshape(-1, 1, 2)], False, 255,
                  max(1, round(thickness)), cv2.LINE_AA)
    return m


def dilate(m: np.ndarray, r: float) -> np.ndarray:
    r = max(0, round(r))
    if r == 0:
        return m
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))


def erode(m: np.ndarray, r: float) -> np.ndarray:
    r = max(0, round(r))
    if r == 0:
        return m
    return cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))


def feather(m: np.ndarray, sigma: float) -> np.ndarray:
    """0/255 마스크 → float32 0~1, 가장자리를 부드럽게."""
    f = m.astype(np.float32) / 255.0 if m.dtype == np.uint8 else m.astype(np.float32)
    if sigma > 0.3:
        f = cv2.GaussianBlur(f, (0, 0), sigma)
    return f


def face_box(regions: dict[str, np.ndarray], shape: tuple[int, int], margin: float) -> tuple[int, int, int, int]:
    """얼굴 윤곽을 둘러싼 (x0, y0, x1, y1), margin px만큼 넓혀서 이미지 안으로 자른다."""
    oval = regions["얼굴윤곽"]
    h, w = shape[:2]
    x0, y0 = np.floor(oval.min(axis=0) - margin).astype(int)
    x1, y1 = np.ceil(oval.max(axis=0) + margin).astype(int)
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


FEATURES = ("왼눈", "오른눈", "왼눈썹", "오른눈썹", "입술", "코아래")


def features_mask(shape, regions, offset, margin: float) -> np.ndarray:
    """눈·눈썹·입술·콧구멍 둘레를 margin px만큼 넓힌 영역 (0/255)."""
    m = np.zeros(shape[:2], np.uint8)
    for k in FEATURES:
        if k in regions:
            m |= poly_mask(shape, regions[k], hull=True, offset=offset)
    return dilate(m, margin)
