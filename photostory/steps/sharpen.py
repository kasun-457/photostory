"""6. 눈·코·입 선명하게 (가이드 §7, 액션 "수정-02"를 규칙 방식으로 재현).

- 액션: 언샵 마스크 350%, 반경 1.8px, 한계값 0 → 얼굴 부분 레이어 불투명도 80%.
- 흐릴 때만 한다. 흐림 판단은 prepare에서 한 번 잰 값(core/measure/sharpness.py)으로 한다.
  애매하면(기준보다 또렷하면) 하지 않고, 하지 않은 이유를 메모에 남긴다.
- 영역: 눈·눈썹·콧구멍 둘레·콧대·입술. 피부(볼·이마)는 건드리지 않는다.
- 강도 부위는 설계서 3-2 표대로 '피부'. 반경 1.8px은 원본 4472px 기준이라 얼굴 폭 배율로 바꾼다.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core.measure import regions as R
from ..core.measure.sharpness import SOFT
from ..core.pipeline import Frame, Step

AMOUNT = 3.5
RADIUS = 1.8
LAYER_OPACITY = 0.8
FEATURES = ("왼눈", "오른눈", "왼눈썹", "오른눈썹", "입술", "코아래")


class SharpenStep(Step):
    id = "sharpen"
    title = "눈·코·입 선명하게"
    param_keys = ("strength.skin",)
    cache = True

    def run(self, frame: Frame, analysis, params) -> Frame:
        st = float(params.get("strength.skin", 1.0))
        sharp = analysis.extra.get("sharpness") if analysis is not None else None
        regs = frame.regions
        if st <= 0 or "얼굴윤곽" not in regs:
            return frame
        if sharp is None or sharp >= SOFT:
            if sharp is not None:
                frame.notes.append(f"선명하게: 하지 않음 (눈·코·입 또렷함 {sharp:.3f} ≥ {SOFT})")
            return frame
        k = R.face_k(frame.points)
        fw = R.face_w(frame.points)
        x0, y0, x1, y1 = R.face_box(regs, frame.image.shape, 60 * k)
        crop = frame.image[y0:y1, x0:x1]
        off = (x0, y0)
        m = np.zeros(crop.shape[:2], np.uint8)
        for key in FEATURES:
            if key in regs:
                m |= R.poly_mask(crop.shape, regs[key], hull=True, offset=off)
        if "콧대" in regs:
            m |= R.line_mask(crop.shape, regs["콧대"], 0.08 * fw, off)
        alpha = R.feather(R.dilate(m, 25 * k), 15 * k) * min(1.0, LAYER_OPACITY * st)
        o = crop.astype(np.float32)
        blur = cv2.GaussianBlur(o, (0, 0), max(0.5, RADIUS * k))
        sharpened = o + AMOUNT * (o - blur)                    # 한계값 0
        out = o + alpha[..., None] * (sharpened - o)
        img = frame.image.copy()
        img[y0:y1, x0:x1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        frame.image = img
        frame.notes.append(f"선명하게: 함 (눈·코·입 또렷함 {sharp:.3f} < {SOFT})")
        return frame
