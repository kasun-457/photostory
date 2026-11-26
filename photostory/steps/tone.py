"""7. 닷징·버닝 (가이드 §8, 액션 "닷징버닝"을 규칙 방식으로 재현).

액션이 만드는 스냅숏:
  스냅숏 2 = 곱하기 모드 병합 (어둡게, 쉐도우)
  스냅숏 3 = 스크린 모드 병합 (밝게, 하이라이트)
사람은 작업 내역 브러시로
  - 하이라이트: 콧대 세로 줄을 스냅숏 3으로 60%
  - 쉐도우: 코 옆과 얼굴 윤곽(턱선)의 어두운 부분을 스냅숏 2로 '과하지 않게'
칠한 뒤 레이어 불투명도로 전체 강도를 다듬는다.

- 강도(명암)는 전체 레이어 불투명도에 곱한다.
- 레이어 불투명도·쉐도우 브러시 값은 가이드에 숫자가 없다. 보수적으로 낮게 잡은 값이며
  실제 사진관 결과와 비교해 맞춰야 한다 (확실하지 않음).
- 콧볼 옆 볼 그림자 지우기(복제 도장)는 아직 없다.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core.measure import regions as R
from ..core.pipeline import Frame, Step

HIGHLIGHT_OPACITY = 0.6     # 가이드: 콧대 60%
SHADOW_OPACITY = 0.35       # 가이드: '낮은 불투명도에서 시작' (숫자 없음)
LAYER_OPACITY = 0.5         # 가이드: 마스터 불투명도로 다듬음 (숫자 없음)


def multiply(o: np.ndarray) -> np.ndarray:
    return o * o / 255.0


def screen(o: np.ndarray) -> np.ndarray:
    return 255.0 - (255.0 - o) * (255.0 - o) / 255.0


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


class ToneStep(Step):
    id = "tone"
    title = "명암 (닷징·버닝)"
    param_keys = ("strength.tone",)
    cache = True

    def masks(self, frame: Frame, shape, off) -> tuple[np.ndarray, np.ndarray]:
        """(하이라이트, 쉐도우) 브러시 마스크 float32 0~1. 각자 최대값 = 브러시 불투명도."""
        regs, p = frame.regions, frame.points
        k = R.face_k(p)
        fw = R.face_w(p)
        hi = np.zeros(shape[:2], np.float32)
        sh = np.zeros(shape[:2], np.float32)

        # 하이라이트: 콧대 (미간 아래 ~ 코끝 위). 코끝 번들거림은 피하려고 끝 점은 뺀다
        if "콧대" in regs and len(regs["콧대"]) >= 4:
            line = regs["콧대"][1:-1]
            band = R.line_mask(shape, line, 0.045 * fw, off)
            hi = R.feather(band, 0.02 * fw) * HIGHLIGHT_OPACITY

        # 쉐도우 1: 코 옆 (콧대 옆선 → 콧볼 위)
        shadow = np.zeros(shape[:2], np.uint8)
        top = p.get("코끝")
        bridge = regs["콧대"][1] if "콧대" in regs else None
        for side, inner, wing in (("왼", "왼눈_안쪽꼭짓점", "왼콧볼"), ("오른", "오른눈_안쪽꼭짓점", "오른콧볼")):
            if bridge is None or inner not in p or wing not in p or top is None:
                continue
            a = _lerp(tuple(bridge), p[inner], 0.55)
            b = _lerp(p[wing], (p[wing][0], a[1]), 0.15)
            shadow |= R.line_mask(shape, np.array([a, b], np.float32), 0.04 * fw, off)

        # 쉐도우 2: 얼굴 윤곽 (턱선 꺾은선 둘레, 얼굴 안쪽 위주)
        jaw = np.zeros(shape[:2], np.uint8)
        for key in ("왼턱선_선", "오른턱선_선"):
            if key in regs:
                jaw |= R.line_mask(shape, regs[key], 0.07 * fw, off)
        if "얼굴윤곽" in regs:
            jaw &= R.dilate(R.poly_mask(shape, regs["얼굴윤곽"], offset=off), 0.02 * fw)
        shadow |= jaw
        # 눈·입·콧구멍은 건드리지 않는다
        shadow &= cv2.bitwise_not(R.features_mask(shape, regs, off, 20 * k))
        sh = R.feather(shadow, 0.025 * fw) * SHADOW_OPACITY
        return hi, sh

    def run(self, frame: Frame, analysis, params) -> Frame:
        st = float(params.get("strength.tone", 1.0))
        regs = frame.regions
        if st <= 0 or "얼굴윤곽" not in regs:
            return frame
        k = R.face_k(frame.points)
        x0, y0, x1, y1 = R.face_box(regs, frame.image.shape, 120 * k)
        crop = frame.image[y0:y1, x0:x1]
        hi, sh = self.masks(frame, crop.shape, (x0, y0))
        if frame.person_mask is not None:
            pm = frame.person_mask[y0:y1, x0:x1]
            hi, sh = hi * pm, sh * pm
        layer = min(1.0, LAYER_OPACITY * st)
        o = crop.astype(np.float32)
        out = o + (hi * layer)[..., None] * (screen(o) - o)
        out = out + (sh * layer)[..., None] * (multiply(out) - out)
        img = frame.image.copy()
        img[y0:y1, x0:x1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        frame.image = img
        return frame
