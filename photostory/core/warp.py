"""변형 엔진 (2단계: 대칭·목·어깨에서 공용).

가이드의 두 가지 손 작업을 같은 식으로 다룬다.
- 픽셀 유동화(§2-5): 넓은 범위를 부드럽게 민다      → 가우시안 같은 무게 지도 × 이동
- 부분 이동(§2-6): 올가미로 오려 옮기거나 돌린다     → 페더 준 조각 마스크 × 아핀 변환

동작 하나 = (무게 지도 w, 아핀 A). 내용물이 p → p + w(p)·(A·p − p) 로 움직인다.
동작을 여러 개 쌓아도 '출력 화소 → 원본 위치' 지도만 합성해 두었다가 마지막에 한 번만 샘플링한다
(여러 번 샘플링해서 흐려지는 것을 막는다).

- 작업 범위(ROI) 안에서만 지도를 만든다.
- 지도는 부드러워서 성긴 격자(grid px마다 한 점)로 계산하고, 마지막에 한 번 키워서 쓴다.
  원본 해상도(4472×6708)의 몸 전체 같은 넓은 범위도 메모리·시간이 적게 든다.
- 무게 지도는 격자 크기(roi_shape)로 만든다. 좌표는 to_local()로 격자 좌표로 바꿔서 그린다.
- 점·영역 다각형은 같은 식으로 앞으로 옮기고, 마스크(사람·머리카락·피부)는 같은 지도로 옮긴다.
"""
from __future__ import annotations

import cv2
import numpy as np

ITER = 3   # 역방향 위치를 찾는 고정점 반복 횟수 (이동이 작아서 3번이면 충분)


class Warp:
    def __init__(self, shape: tuple[int, int], roi: tuple[int, int, int, int], grid: int = 1):
        h, w = shape[:2]
        x0, y0, x1, y1 = roi
        self.x0, self.y0 = max(0, int(x0)), max(0, int(y0))
        self.x1, self.y1 = min(w, int(x1)), min(h, int(y1))
        self.g = max(1, int(grid))
        self.shape = (h, w)
        hc = -(-(self.y1 - self.y0) // self.g)
        wc = -(-(self.x1 - self.x0) // self.g)
        # 격자 점 = 칸의 가운데 (cv2.resize로 키울 때 화소 중심이 맞도록)
        c = (self.g - 1) / 2
        gx, gy = np.meshgrid(self.x0 + c + self.g * np.arange(wc, dtype=np.float32),
                             self.y0 + c + self.g * np.arange(hc, dtype=np.float32))
        self.gx, self.gy = gx.astype(np.float32), gy.astype(np.float32)
        self.mx, self.my = self.gx.copy(), self.gy.copy()      # 출력 → 원본 위치 (누적)
        self.ops: list[tuple[np.ndarray, np.ndarray]] = []
        self.moved = False

    # ---------------------------------------------------------------- 좌표
    @property
    def roi_shape(self) -> tuple[int, int]:
        """무게 지도의 크기 (격자)."""
        return self.gx.shape

    def to_local(self, pts) -> np.ndarray:
        """전체 이미지 좌표 → 격자 좌표 (무게 지도에 그릴 때)."""
        p = np.asarray(pts, np.float32).reshape(-1, 2)
        c = (self.g - 1) / 2
        return np.stack([(p[:, 0] - self.x0 - c) / self.g, (p[:, 1] - self.y0 - c) / self.g], axis=1)

    def px(self, v: float) -> float:
        """이미지 px 길이 → 격자 칸 수."""
        return v / self.g

    def _local_xy(self, x, y):
        c = (self.g - 1) / 2
        return (x - self.x0 - c) / self.g, (y - self.y0 - c) / self.g

    # ---------------------------------------------------------------- 동작
    def add(self, weight: np.ndarray, A: np.ndarray) -> None:
        """weight: roi_shape 크기 float32 0~1 (지금 좌표 기준). A: 2x3 아핀 (전체 이미지 좌표)."""
        A = np.asarray(A, np.float32)
        if weight.shape != self.roi_shape or not np.any(weight > 1e-4):
            return
        w = weight.astype(np.float32)
        # 역방향: s = q − w(s)·(A·s − s) 를 반복으로 푼다 (q = 출력 격자)
        sx, sy = self.gx.copy(), self.gy.copy()
        for _ in range(ITER):
            lx, ly = self._local_xy(sx, sy)
            ws = cv2.remap(w, lx, ly, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            ax = A[0, 0] * sx + A[0, 1] * sy + A[0, 2] - sx
            ay = A[1, 0] * sx + A[1, 1] * sy + A[1, 2] - sy
            sx, sy = self.gx - ws * ax, self.gy - ws * ay
        # 누적 지도 합성: 새 출력 q → s(q) → 이전 지도(s)
        lx, ly = self._local_xy(sx, sy)
        self.mx = cv2.remap(self.mx, lx, ly, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        self.my = cv2.remap(self.my, lx, ly, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
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
            lx, ly = self._local_xy(p[:, 0], p[:, 1])
            inside = (lx >= -0.5) & (ly >= -0.5) & (lx <= w.shape[1] - 0.5) & (ly <= w.shape[0] - 0.5)
            if not inside.any():
                continue
            ww = np.zeros(len(p), np.float32)
            ww[inside] = cv2.remap(w, lx[inside].reshape(-1, 1).astype(np.float32),
                                   ly[inside].reshape(-1, 1).astype(np.float32), cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE).ravel()
            tx = A[0, 0] * p[:, 0] + A[0, 1] * p[:, 1] + A[0, 2]
            ty = A[1, 0] * p[:, 0] + A[1, 1] * p[:, 1] + A[1, 2]
            p[:, 0] += ww * (tx - p[:, 0])
            p[:, 1] += ww * (ty - p[:, 1])
        return p

    # ---------------------------------------------------------------- 적용
    def _full_maps(self) -> tuple[np.ndarray, np.ndarray]:
        h, w = self.y1 - self.y0, self.x1 - self.x0
        if self.g == 1:
            return self.mx, self.my
        hc, wc = self.roi_shape
        size = (wc * self.g, hc * self.g)
        mx = cv2.resize(self.mx, size, interpolation=cv2.INTER_LINEAR)[:h, :w]
        my = cv2.resize(self.my, size, interpolation=cv2.INTER_LINEAR)[:h, :w]
        return mx, my

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        if not self.moved:
            return img
        mx, my = self._full_maps()
        out = img.copy()
        out[self.y0:self.y1, self.x0:self.x1] = cv2.remap(img, mx, my, cv2.INTER_LANCZOS4,
                                                          borderMode=cv2.BORDER_REPLICATE)
        return out

    def apply_mask(self, m: np.ndarray | None) -> np.ndarray | None:
        if m is None or not self.moved:
            return m
        mx, my = self._full_maps()
        out = m.copy()
        out[self.y0:self.y1, self.x0:self.x1] = cv2.remap(m, mx, my, cv2.INTER_LINEAR,
                                                          borderMode=cv2.BORDER_REPLICATE)
        return out

    def apply_frame(self, frame) -> None:
        """Frame의 이미지·마스크·점·영역을 함께 옮긴다 (제자리 수정)."""
        if not self.moved:
            return
        mx, my = self._full_maps()

        def rm(m, interp):
            if m is None:
                return None
            out = m.copy()
            out[self.y0:self.y1, self.x0:self.x1] = cv2.remap(m, mx, my, interp, borderMode=cv2.BORDER_REPLICATE)
            return out

        frame.image = rm(frame.image, cv2.INTER_LANCZOS4)
        frame.person_mask = rm(frame.person_mask, cv2.INTER_LINEAR)
        frame.hair_mask = rm(frame.hair_mask, cv2.INTER_LINEAR)
        frame.skin_mask = rm(frame.skin_mask, cv2.INTER_LINEAR)
        frame.masks = {k: rm(v, cv2.INTER_LINEAR) for k, v in frame.masks.items()}
        names = list(frame.points)
        if names:
            moved = self.forward_points(np.array([frame.points[k] for k in names]))
            frame.points = {k: (float(x), float(y)) for k, (x, y) in zip(names, moved)}
        frame.regions = {k: self.forward_points(v) if len(v) else v for k, v in frame.regions.items()}


def gaussian_weight(warp: Warp, center, sigma_x: float, sigma_y: float | None = None,
                    peak: float = 1.0) -> np.ndarray:
    """유동화 붓: center(이미지 좌표)를 중심으로 한 타원 가우시안 무게 (격자 크기). sigma는 이미지 px."""
    sigma_y = sigma_x if sigma_y is None else sigma_y
    gx, gy = warp.gx[0], warp.gy[:, 0]
    wx = np.exp(-0.5 * ((gx - center[0]) / max(sigma_x, 1e-3)) ** 2)
    wy = np.exp(-0.5 * ((gy - center[1]) / max(sigma_y, 1e-3)) ** 2)
    return (wy[:, None] * wx[None, :] * peak).astype(np.float32)
