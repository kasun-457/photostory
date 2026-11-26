"""12. 안경 얹기 (설계서 3-2, v1.52 '안경 앉히기').

prepare(P5, core/glasses.py)에서 원본 좌표로 떼어 둔 안경테 조각을 지금 사진(수평·대칭이 끝난 뒤)에 얹는다.
- 원본 → 지금 좌표: 렌즈와 상관없는 얼굴 점 + 원본의 눈 꼭짓점으로 닮음 변환 (수평 회전·대칭 이동 반영)
- 앉히는 기준 (v1.52):
  · 두 렌즈 아래가 수평 → 안경테 조각을 돌린다
  · 안경 바깥 끝이 중심선에서 같은 거리 → 옆으로 옮긴다
  · 안경테가 윗눈꺼풀(눈 안)을 가리면 조금(얼굴 폭 3%까지) 올린다
- 이 변환들을 하나로 합쳐 조각을 한 번만 샘플링해 얹는다.
- 렌즈 반사·렌즈로 커지거나 작아진 눈은 가져오지 않는다 (원본 눈이 그대로).
- 복원 붓 (v1.52 '안경수정'): 결과 화면에서 칠한 곳을 고친다.
  · 왼쪽 버튼 = 안경 쓴 사진에서 살림 (자동으로 못 뗀 은색 힌지·끊긴 테)
  · 오른쪽 버튼 = 원본으로 되돌림 (테와 함께 딸려 온 머리카락·배경)
  획은 수정본 크기 대비 비율 좌표라 미리보기·원본 해상도에서 같은 곳을 칠한다.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..core.glasses import STABLE, _compose
from ..core.measure import regions as R
from ..core.pipeline import Frame, Step

LIFT_MAX = 0.03      # 얼굴 폭 대비


def stroke_mask(shape, radius: float, pts) -> np.ndarray:
    """붓 획 하나(비율 좌표) → float32 0~1 마스크. 가장자리는 반지름의 15%만큼 부드럽게."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    r = max(1, round(radius * w))
    xy = np.round(np.asarray(pts, np.float64).reshape(-1, 2) * [w, h]).astype(np.int32)
    if len(xy) > 1:
        cv2.polylines(m, [xy.reshape(-1, 1, 2)], False, 255, 2 * r, cv2.LINE_AA)
    for q in (xy[0], xy[-1]):
        cv2.circle(m, (int(q[0]), int(q[1])), r, 255, -1, cv2.LINE_AA)
    return R.feather(m, 0.15 * r)


def _warp_alpha(alpha, T, shape):
    return cv2.warpAffine(alpha, T, (shape[1], shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _lens_bottoms(a: np.ndarray, cx: float):
    """왼쪽·오른쪽 반에서 테가 있는 가장 아래 점 (x, y)."""
    out = []
    for sl in (slice(0, int(cx)), slice(int(cx), a.shape[1])):
        ys, xs = np.nonzero(a[:, sl] > 0.5)
        if ys.size == 0:
            return None
        y = ys.max()
        out.append((float(xs[ys >= y - 1].mean() + sl.start), float(y)))
    return out


class GlassesStep(Step):
    id = "glasses"
    title = "안경 얹기"
    param_keys = ("glasses", "glasses_strokes")
    cache = True

    def run(self, frame: Frame, analysis, params) -> Frame:
        if not params.get("glasses") or analysis is None:
            return frame
        cut = analysis.extra.get("glasses")
        if cut is None:
            note = analysis.extra.get("glasses_note") or "안경: 안경 쓴 사진이 없어 얹지 않음"
            frame.notes.append(note)
            return frame
        p0, p = analysis.points, frame.points
        keys = [k for k in STABLE + ("왼눈_바깥꼭짓점", "오른눈_바깥꼭짓점", "왼눈_안쪽꼭짓점", "오른눈_안쪽꼭짓점")
                if k in p0 and k in p]
        if len(keys) < 6:
            return frame
        S, _ = cv2.estimateAffinePartial2D(np.float32([p0[k] for k in keys]), np.float32([p[k] for k in keys]),
                                           method=cv2.LMEDS)
        if S is None:
            return frame
        ox, oy = cut["offset"]
        T = _compose(S, np.float32([[1, 0, ox], [0, 1, oy]]))           # 조각 → 지금 좌표
        shape = frame.image.shape
        fw = R.face_w(p)
        cx = (p["왼눈_안쪽꼭짓점"][0] + p["오른눈_안쪽꼭짓점"][0]) / 2
        eye_y = (p["왼눈_바깥꼭짓점"][1] + p["오른눈_바깥꼭짓점"][1]) / 2
        a = _warp_alpha(cut["alpha"], T, shape)
        done = []
        # 1. 두 렌즈 아래 수평
        b = _lens_bottoms(a, cx)
        if b is not None:
            (xl, yl), (xr, yr) = b
            deg = math.degrees(math.atan2(yr - yl, xr - xl))
            if abs(deg) >= 0.1:
                Rm = cv2.getRotationMatrix2D((cx, eye_y), deg, 1.0)      # 반대로 돌려 수평
                T = _compose(Rm.astype(np.float32), T)
                a = _warp_alpha(cut["alpha"], T, shape)
                done.append(f"기울기 {-deg:+.1f}°")
        # 2. 바깥 끝이 중심선에서 같은 거리
        cols = np.nonzero((a > 0.5).any(axis=0))[0]
        if cols.size:
            dx = cx - (cols[0] + cols[-1]) / 2
            if abs(dx) >= 0.5:
                T = _compose(np.float32([[1, 0, dx], [0, 1, 0]]), T)
                a = _warp_alpha(cut["alpha"], T, shape)
                done.append(f"가운데 {dx / frame.scale:+.0f}px")
        # 3. 테가 윗눈꺼풀을 가리면 조금 올림
        eyes = np.zeros(shape[:2], np.uint8)
        for k in ("왼눈", "오른눈"):
            if k in frame.regions:
                eyes |= R.poly_mask(shape, frame.regions[k], hull=True)
        over = (a > 0.5) & (eyes > 0)
        if over.any():
            rows = np.nonzero(over.any(axis=1))[0]
            lift = min(float(rows[-1] - rows[0] + 1), LIFT_MAX * fw)
            T = _compose(np.float32([[1, 0, 0], [0, 1, -lift]]), T)
            done.append(f"올림 {lift / frame.scale:.0f}px")
        # 한 번만 샘플링해서 얹기
        h, w = shape[:2]
        rgb = cv2.warpAffine(cut["rgb"], T, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        a = _warp_alpha(cut["alpha"], T, shape)
        strokes = params.get("glasses_strokes") or []
        if strokes:
            # 살리기는 안경 사진이 있는 띠 안에서만 (띠 밖은 가져올 화소가 없음)
            valid = _warp_alpha(np.ones(cut["alpha"].shape, np.float32), T, shape)
            for mode, radius, pts in strokes:                 # 칠한 순서대로 (나중 획이 이김)
                if not pts:
                    continue
                m = stroke_mask(shape, radius, pts)
                a = np.maximum(a, m * valid) if mode == "glasses" else a * (1.0 - m)
            done.append(f"복원 붓 {len(strokes)}획")
        a = a[..., None]
        out = frame.image.astype(np.float32) * (1 - a) + rgb.astype(np.float32) * a
        frame.image = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        frame.outputs["glasses"] = {"moved": done, "transform": T.tolist()}
        frame.notes.append("안경: 테 얹음" + (f" ({', '.join(done)})" if done else ""))
        return frame
