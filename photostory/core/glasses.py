"""P5 안경 짝 정렬 (설계서 3-1, v1.52 '안경 옮기기').

손님이 안경을 쓰면 두 장을 찍는다: 원본(안경 벗음) + 안경 쓴 사진.
보정은 원본으로 하고, 안경 쓴 사진에서 안경테만 떼어 12번 단계에서 얹는다.
렌즈 반사·렌즈 때문에 커지거나 작아진 눈은 가져오지 않는다 (원본 눈이 그대로 보임).

1. 정렬: 렌즈에 가리지 않는 얼굴 점(코·입·턱·볼·이마·얼굴 옆)으로 닮음 변환을 먼저 잡고,
   안경 띠를 가린 채로 영상 맞춤(ECC)으로 다듬는다. 눈 점은 렌즈가 굴절시켜 쓰지 않는다.
2. 안경테 떼기: 안경 띠(눈썹 위 ~ 코 밑, 얼굴 옆까지) 안에서, 맞춘 안경 사진이 원본과 뚜렷이 다른 곳.
   눈 안(렌즈 너머로 굴절된 눈)은 뺀다. 분할 모델의 '장식' 영역이 있으면 함께 쓴다.
   작은 점(정렬 오차로 생긴 경계)은 버리고 큰 덩어리(테)만 남긴다.
결과는 원본 좌표의 조각(색·알파·위치)으로 Analysis.extra["glasses"]에 둔다.
기준값은 합성 사진으로 잡은 값이다 (확실하지 않음).
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .measure import regions as R

log = logging.getLogger(__name__)

STABLE = ("코끝", "코밑", "왼입꼬리", "오른입꼬리", "윗입술_중심", "아랫입술_중심", "턱끝",
          "왼볼_입높이", "오른볼_입높이", "왼턱선", "오른턱선", "이마_위", "왼얼굴_옆", "오른얼굴_옆")
DIFF_THR = 22.0        # Lab 거리. 이보다 다르면 안경테 후보
WORK_FACE_W = 400.0    # 정렬 다듬기는 얼굴 폭을 이만큼 줄여서


def band_box(points, fw: float, shape) -> tuple[int, int, int, int]:
    """안경이 있을 만한 띠 (x0, y0, x1, y1)."""
    h, w = shape[:2]
    top = min(points.get("왼눈썹_위", points["왼눈_윗꺼풀"])[1], points.get("오른눈썹_위", points["오른눈_윗꺼풀"])[1])
    y0 = top - 0.10 * fw
    y1 = points["코밑"][1] + 0.02 * fw
    lx = points.get("왼얼굴_옆", (points["왼눈_바깥꼭짓점"][0] - 0.25 * fw, 0))[0]
    rx = points.get("오른얼굴_옆", (points["오른눈_바깥꼭짓점"][0] + 0.25 * fw, 0))[0]
    x0, x1 = lx - 0.06 * fw, rx + 0.06 * fw
    return max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))


def align(glass_rgb, glass_pts, orig_rgb, orig_pts) -> tuple[np.ndarray, float] | None:
    """안경 사진 좌표 → 원본 좌표 2x3. (변환, 맞춤 점수 0~1)"""
    keys = [k for k in STABLE if k in glass_pts and k in orig_pts]
    if len(keys) < 6:
        return None
    src = np.float32([glass_pts[k] for k in keys])
    dst = np.float32([orig_pts[k] for k in keys])
    M0, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M0 is None:
        return None
    fw = R.face_w(orig_pts)
    s = min(1.0, WORK_FACE_W / max(fw, 1.0))
    # 원본 얼굴 둘레를 작게 잘라 그 좌표에서 맞춘다
    xs, ys = dst[:, 0], dst[:, 1]
    x0, y0 = int(max(0, xs.min() - 0.5 * fw)), int(max(0, ys.min() - 0.6 * fw))
    x1, y1 = int(min(orig_rgb.shape[1], xs.max() + 0.5 * fw)), int(min(orig_rgb.shape[0], ys.max() + 0.3 * fw))
    C = np.float32([[s, 0, -s * x0], [0, s, -s * y0]])                  # 원본 → 작업 좌표
    size = (max(8, int((x1 - x0) * s)), max(8, int((y1 - y0) * s)))
    tmpl = cv2.cvtColor(cv2.warpAffine(orig_rgb, C, size, flags=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    M0w = _compose(C, M0)                                                 # 안경 → 작업 좌표
    inp = cv2.cvtColor(cv2.warpAffine(glass_rgb, M0w, size, flags=cv2.INTER_LINEAR), cv2.COLOR_RGB2GRAY)
    mask = np.full(tmpl.shape, 255, np.uint8)
    bx0, by0, bx1, by1 = band_box(orig_pts, fw, orig_rgb.shape)
    cv2.rectangle(mask, (int((bx0 - x0) * s), int((by0 - y0) * s)), (int((bx1 - x0) * s), int((by1 - y0) * s)), 0, -1)
    warp = np.eye(2, 3, dtype=np.float32)
    score = 0.0
    try:
        score, warp = cv2.findTransformECC(tmpl.astype(np.float32), inp.astype(np.float32), warp,
                                           cv2.MOTION_AFFINE,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
                                           mask, 5)
    except cv2.error as e:   # 맞추지 못하면 얼굴 점 변환만 쓴다
        log.info("ECC 실패: %s", e)
        return M0.astype(np.float32), 0.0
    # 작업 좌표에서: inp(x) ≈ tmpl(W·x) → 안경 → 원본 = C⁻¹ · W⁻¹ · C · M0
    Winv = cv2.invertAffineTransform(warp)
    Cinv = cv2.invertAffineTransform(C)
    M = _compose(Cinv, _compose(Winv, _compose(C, M0)))
    return M.astype(np.float32), float(score)


def _compose(A, B) -> np.ndarray:
    """A∘B (먼저 B, 다음 A). 둘 다 2x3."""
    A3 = np.vstack([A, [0, 0, 1]])
    B3 = np.vstack([B, [0, 0, 1]])
    return (A3 @ B3)[:2].astype(np.float32)


def extract(orig_rgb, glass_aligned, orig_pts, orig_regions, accessory=None) -> dict | None:
    """원본 좌표로 맞춘 안경 사진에서 안경테 조각을 뗀다."""
    fw = R.face_w(orig_pts)
    x0, y0, x1, y1 = band_box(orig_pts, fw, orig_rgb.shape)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    a = cv2.cvtColor(orig_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2LAB).astype(np.float32)
    g = cv2.cvtColor(glass_aligned[y0:y1, x0:x1], cv2.COLOR_RGB2LAB).astype(np.float32)
    sig = max(0.6, 0.003 * fw)
    d = np.linalg.norm(cv2.GaussianBlur(g, (0, 0), sig) - cv2.GaussianBlur(a, (0, 0), sig), axis=2)
    cand = d > DIFF_THR
    if accessory is not None:
        acc = accessory[y0:y1, x0:x1] > 0.5
        cand |= acc & (d > 0.5 * DIFF_THR)
    # 눈 안(렌즈 너머 굴절)은 빼기
    eyes = np.zeros(cand.shape, np.uint8)
    for k in ("왼눈", "오른눈"):
        if k in orig_regions:
            eyes |= R.poly_mask(cand.shape, orig_regions[k], hull=True, offset=(x0, y0))
    cand &= R.dilate(eyes, 0.01 * fw) == 0
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if n < 2:
        return None
    big = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 0.004 * fw * fw
           and st[i, cv2.CC_STAT_WIDTH] >= 0.15 * fw]
    if not big:
        return None
    m = np.isin(lab, big).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    alpha = R.feather(m, max(0.5, 0.0015 * fw))
    return {"rgb": glass_aligned[y0:y1, x0:x1].copy(), "alpha": alpha.astype(np.float32),
            "offset": (x0, y0), "area": int((m > 0).sum())}


def prepare_glasses(orig, glass) -> dict | None:
    """두 Analysis로 안경테 조각을 만든다 (P5). 실패하면 None."""
    al = align(glass.image, glass.points, orig.image, orig.points)
    if al is None:
        return None
    M, score = al
    h, w = orig.image.shape[:2]
    aligned = cv2.warpAffine(glass.image, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    acc = glass.masks.get("장식")
    acc_al = None if acc is None else cv2.warpAffine(acc, M, (w, h), flags=cv2.INTER_LINEAR)
    cut = extract(orig.image, aligned, orig.points, orig.regions, acc_al)
    if cut is None:
        return None
    cut["align_score"] = round(score, 3)
    return cut
