"""변형 엔진 (2단계: 대칭·목·어깨에서 공용).

가이드의 두 가지 손 작업을 같은 식으로 다룬다.
- 픽셀 유동화(§2-5): 넓은 범위를 부드럽게 민다      → 가우시안 같은 무게 지도 × 이동
- 부분 이동(§2-6): 올가미로 오려 옮기거나 돌린다     → 페더 준 조각 마스크 × 아핀 변환

동작 하나 = (무게 지도 w, 아핀 A). 내용물이 p → p + w(p)·(A·p − p) 로 움직인다.
동작을 여러 개 쌓아도 '출력 화소 → 원본 위치' 지도만 합성해 두었다가 마지막에 한 번만 샘플링한다
(v1.52처럼 여러 번 샘플링해서 흐려지는 것을 막는다).

- 작업 범위(ROI) 안에서만 지도를 만든다 (원본 해상도에서 메모리 절약).
- 점·영역 다각형은 같은 식으로 앞으로 옮긴다. 마스크(사람·머리카락·피부)는 같은 지도로 옮긴다.
"""
from __future__ import annotations

import cv2
import numpy as np

ITER = 3   # 역방향 위치를 찾는 고정점 반복 횟수 (이동이 작아서 3번이면 충분)


class Warp:
    def __init__(self, shape: tuple[int, int], roi: tuple[int, int, int, int]):
        h, w = shape[:2]
        x0, y0, x1, y1 = roi
        self.x0, self.y0 = max(0, int(x0)), max(0, int(y0))
        self.x1, self.y1 = min(w, int(x1)), min(h, int(y1))
        self.shape = (h, w)
        gx, gy = np.meshgrid(np.arange(self.x0, self.x1, dtype=np.float32),
                             np.arange(self.y0, self.y1, dtype=np.float32))
        self.gx, self.gy = gx, gy
        self.mx, self.my = gx.copy(), gy.copy()       # 출력 화소 → 원본 위치 (누적)
        self.ops: list[tuple[np.ndarray, np.ndarray]] = []
        self.moved = False

    @property
    def roi_shape(self) -> tuple[int, int]:
        return self.y1 - self.y0, self.x1 - self.x0

    def offset(self) -> tuple[int, int]:
        return self.x0, self.y0

    # ---------------------------------------------------------------- 동작
    def add(self, weight: np.ndarray, A: np.ndarray) -> None:
        """weight: ROI 크기 float32 0~1 (지금 좌표 기준). A: 2x3 아핀 (전체 이미지 좌표)."""
        A = np.asarray(A, np.float32)
        if weight.shape != self.roi_shape or not np.any(weight > 1e-4):
            return
        w = weight.astype(np.float32)
        # 역방향: s = q − w(s)·(A·s − s) 를 반복으로 푼다 (q = 출력 격자)
        sx, sy = self.gx.copy(), self.gy.copy()
        for _ in range(ITER):
            ws = cv2.remap(w, sx - self.x0, sy - self.y0, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            ax = A[0, 0] * sx + A[0, 1] * sy + A[0, 2] - sx
            ay = A[1, 0] * sx + A[1, 1] * sy + A[1, 2] - sy
            sx, sy = self.gx - ws * ax, self.gy - ws * ay
        # 누적 지도 합성: 새 출력 q → s(q) → 이전 지도(s)
        self.mx = cv2.remap(self.mx, sx - self.x0, sy - self.y0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        self.my = cv2.remap(self.my, sx - self.x0, sy - self.y0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        self.ops.append((w, A))
        self.moved = True

    def shift(self, weight: np.ndarray, dx: float, dy: float) -> None:
        self.add(weight, np.float32([[1, 0, dx], [0, 1, dy]]))

    def rotate(self, weight: np.ndarray, center: tuple[float, float], deg: float) -> None:
        """사진 기준 시계 방향 +deg (y가 아래인 좌표에서 회전)."""
        self.add(weight, cv2.getRotationMatrix2D((float(center[0]), float(center[1])), -deg, 1.0))

    # ---------------------------------------------------------------- 점 옮기기
    def forward_points(self, pts: np.ndarray, ops=None) -> np.ndarray:
        """점들을 동작 순서대로 앞으로 옮긴다. pts: (N, 2) 전체 좌표."""
        p = np.asarray(pts, np.float32).reshape(-1, 2).copy()
        for w, A in (self.ops if ops is None else ops):
            lx, ly = p[:, 0] - self.x0, p[:, 1] - self.y0
            inside = (lx >= 0) & (ly >= 0) & (lx <= w.shape[1] - 1) & (ly <= w.shape[0] - 1)
            if not inside.any():
                continue
            ww = np.zeros(len(p), np.float32)
            ww[inside] = cv2.remap(w, lx[inside].reshape(-1, 1), ly[inside].reshape(-1, 1),
                                   cv2.INTER_LINEAR).ravel()
            tx = A[0, 0] * p[:, 0] + A[0, 1] * p[:, 1] + A[0, 2]
            ty = A[1, 0] * p[:, 0] + A[1, 1] * p[:, 1] + A[1, 2]
            p[:, 0] += ww * (tx - p[:, 0])
            p[:, 1] += ww * (ty - p[:, 1])
        return p

    # ---------------------------------------------------------------- 적용
    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if not self.moved:
            return img
        out = img.copy()
        roi = cv2.remap(img, self.mx, self.my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        out[self.y0:self.y1, self.x0:self.x1] = roi
        return out

    def apply_mask(self, m: np.ndarray | None) -> np.ndarray | None:
        if m is None or not self.moved:
            return m
        out = m.copy()
        out[self.y0:self.y1, self.x0:self.x1] = cv2.remap(m, self.mx, self.my, cv2.INTER_LINEAR,
                                                          borderMode=cv2.BORDER_REPLICATE)
        return out

    def apply_frame(self, frame) -> None:
        """Frame의 이미지·마스크·점·영역을 함께 옮긴다 (제자리 수정)."""
        if not self.moved:
            return
        frame.image = self.apply_image(frame.image)
        frame.person_mask = self.apply_mask(frame.person_mask)
        frame.hair_mask = self.apply_mask(frame.hair_mask)
        frame.skin_mask = self.apply_mask(frame.skin_mask)
        frame.masks = {k: self.apply_mask(v) for k, v in frame.masks.items()}
        names = list(frame.points)
        if names:
            moved = self.forward_points(np.array([frame.points[k] for k in names]))
            frame.points = {k: (float(x), float(y)) for k, (x, y) in zip(names, moved)}
        frame.regions = {k: self.forward_points(v) if len(v) else v for k, v in frame.regions.items()}


def gaussian_weight(shape, offset, center, sigma_x: float, sigma_y: float | None = None,
                    peak: float = 1.0) -> np.ndarray:
    """유동화 붓: center를 중심으로 한 타원 가우시안 무게 (ROI 크기)."""
    sigma_y = sigma_x if sigma_y is None else sigma_y
    h, w = shape[:2]
    xs = np.arange(w, dtype=np.float32) + offset[0] - center[0]
    ys = np.arange(h, dtype=np.float32) + offset[1] - center[1]
    gx = np.exp(-0.5 * (xs / max(sigma_x, 1e-3)) ** 2)
    gy = np.exp(-0.5 * (ys / max(sigma_y, 1e-3)) ** 2)
    return (gy[:, None] * gx[None, :] * peak).astype(np.float32)
