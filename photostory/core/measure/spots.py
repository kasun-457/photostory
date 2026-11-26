"""피부 위 작은 점 찾기 (가이드 §6: 트러블은 지우고, 확실한 얼굴 점은 남기고, 애매하면 남긴다).

prepare에서 사진마다 한 번, 얼굴 폭을 일정하게 맞춘 해상도에서 찾는다.
그래서 미리보기와 원본 해상도에서 같은 점을 남긴다.

판정 (규칙 방식, 학습 모델 전까지):
- 주변 피부(중간값+흐림)보다 어두운 작은 덩어리만 후보로 본다. 지름 약 1~8mm.
- 붉은 기가 분명하고 아주 진하지는 않으면 '트러블' → 지운다.
- 나머지(갈색·검은 점, 판단이 애매한 것)는 남긴다. 남긴 위치는 결과 화면 메모에 적는다.
기준값은 공개 사진 몇 장으로 잡은 값이다. 실제 사진관 샘플로 다시 맞춰야 한다 (확실하지 않음).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import regions as R

WORK_FACE_W = 600.0       # 찾을 때 얼굴 폭을 이 px로 맞춘다
MIN_DIAM_MM, MAX_DIAM_MM = 1.0, 8.0
FACE_W_MM = 140.0         # 입 높이 얼굴 폭 (성인 평균 근사)
DARK_L = 10.0             # Lab L(0~255)에서 주변보다 이만큼 어두우면 후보
TROUBLE_REDNESS = 5.0     # 주변보다 a*가 이만큼 높으면 붉은 기
MOLE_CONTRAST = 35.0      # 이보다 진하면 붉어도 점으로 보고 남긴다
MIN_CONTRAST = 15.0       # 덩어리 평균이 이보다 옅으면 점이 아니라 피부 결


@dataclass
class Spot:
    x: float
    y: float
    r: float          # 반지름, 입력 이미지 px
    kind: str         # "트러블" | "점" | "애매"
    contrast: float   # 주변보다 어두운 정도 (L)
    redness: float    # 주변보다 붉은 정도 (a*)

    @property
    def keep(self) -> bool:
        return self.kind != "트러블"


def skin_area(shape, regions, skin_mask, offset, k) -> np.ndarray:
    """칠할 피부 영역 (0/255): 얼굴 윤곽 안 ∩ 피부 분할 − 눈·눈썹·입·콧구멍 둘레."""
    oval = R.poly_mask(shape, regions["얼굴윤곽"], offset=offset)
    feat = R.features_mask(shape, regions, offset, 45 * k)
    m = cv2.bitwise_and(oval, cv2.bitwise_not(feat))
    if skin_mask is not None:
        m = cv2.bitwise_and(m, np.where(skin_mask > 0.5, 255, 0).astype(np.uint8))
    return m


def smooth_base(img: np.ndarray, k: float) -> np.ndarray:
    """가이드 '수정-01'의 앞부분: 중간값 25px → 가우시안 흐림 8px.

    결과가 아주 부드러워서 작게 줄여 계산한 뒤 키워도 차이가 거의 없다 (속도).
    """
    h, w = img.shape[:2]
    r_med = 25.0 * k
    d = min(1.0, 6.0 / max(r_med, 1e-3))        # 줄였을 때 중간값 반지름이 6px 정도가 되게
    small = cv2.resize(img, (max(1, round(w * d)), max(1, round(h * d))), interpolation=cv2.INTER_AREA) if d < 1 else img
    ks = max(3, 2 * round(r_med * d) + 1)
    out = cv2.medianBlur(small, ks)
    out = cv2.GaussianBlur(out, (0, 0), max(0.5, 8.0 * k * d))
    if d < 1:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def find_spots(rgb: np.ndarray, points, regions, skin_mask=None) -> list[Spot]:
    if "얼굴윤곽" not in regions:
        return []
    fw = R.face_w(points)
    s = min(1.0, WORK_FACE_W / max(fw, 1.0))
    x0, y0, x1, y1 = R.face_box(regions, rgb.shape, 0.05 * fw)
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return []
    if s < 1:
        crop = cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))),
                          interpolation=cv2.INTER_AREA)
    sm = None
    if skin_mask is not None:
        sm = cv2.resize(skin_mask[y0:y1, x0:x1], (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
    regs = {k: (v - np.array([x0, y0], np.float32)) * s for k, v in regions.items()}
    pts = {k: ((x - x0) * s, (y - y0) * s) for k, (x, y) in points.items()}
    k = R.face_k(pts)

    area = skin_area(crop.shape, regs, sm, (0, 0), k)
    area = R.erode(area, 4)                      # 영역 가장자리(그림자 경계)에서 생기는 가짜 점 줄이기
    base = smooth_base(crop, k)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(base, cv2.COLOR_RGB2LAB).astype(np.float32)
    dL = lab[..., 0] - lab_b[..., 0]
    da = lab[..., 1] - lab_b[..., 1]

    cand = ((dL < -DARK_L) & (area > 0)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab_img, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
    px_per_mm = R.face_w(pts) / FACE_W_MM
    dmin, dmax = MIN_DIAM_MM * px_per_mm, MAX_DIAM_MM * px_per_mm
    out: list[Spot] = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        diam = max(w, h)
        if diam < dmin or diam > dmax:
            continue
        if a < 0.35 * (np.pi * (diam / 2) ** 2):  # 길쭉한 것(주름·머리카락·그림자 선)은 점이 아님
            continue
        sel = lab_img[y:y + h, x:x + w] == i
        contrast = float(-dL[y:y + h, x:x + w][sel].mean())
        redness = float(da[y:y + h, x:x + w][sel].mean())
        if contrast < MIN_CONTRAST:                # 또렷하지 않은 것은 피부 결로 보고 함께 정리한다
            continue
        if redness >= TROUBLE_REDNESS and contrast < MOLE_CONTRAST:
            kind = "트러블"
        elif contrast >= 2 * DARK_L and redness < TROUBLE_REDNESS / 2:
            kind = "점"
        else:
            kind = "애매"
        cx, cy = cents[i]
        out.append(Spot(float(cx / s + x0), float(cy / s + y0), float(diam / 2 / s), kind,
                        round(contrast, 1), round(redness, 1)))
    return out
