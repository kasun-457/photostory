"""3·4. 피부 잡티 제거와 결 정리 (가이드 §6, 액션 "수정-01"을 규칙 방식으로 재현).

액션이 만드는 '스냅숏 1':
  레이어 복제 → 중간값 25px → 가우시안 흐림 8px (= A)
  → 복제 → 노이즈 추가 6% 균일(컬러) → 더 흐리게 (= B)
  → 두 레이어 불투명도 80%   ⇒  스냅숏 1 = 0.8·B + 0.2·(0.8·A + 0.2·원본)
사람은 작업 내역 브러시(불투명도 60%, 부드러운 브러시)로 스냅숏 1을 피부에 칠한다.
여기서는 '얼굴 윤곽 − 눈·눈썹·코·입 둘레' 영역을 부드럽게 만들어 60%로 섞는다.

- 강도(피부)는 60%에 곱한다. 0이면 아무것도 안 함.
- prepare에서 남기기로 한 점(확실한 점·애매한 점) 둘레는 칠하지 않는다.
- px 값은 원본 4472px 기준이라 얼굴 폭 배율(face_k)로 바꾼다.
- 학습 모델이 준비되면 이 파일 대신 모델 결과를 쓴다 (설계서 6-2).
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core.measure import regions as R
from ..core.measure.spots import skin_area, smooth_base
from ..core.pipeline import Frame, Step

BRUSH_OPACITY = 0.6
LAYER_OPACITY = 0.8
NOISE = 0.06            # 노이즈 6% 균일
FEATHER = 80.0          # 페더 60~100px의 가운데 (원본 px)
SEED = 20261126         # 같은 사진은 언제나 같은 결과가 나오게


def snapshot1(orig: np.ndarray, k: float, scale: float) -> np.ndarray:
    """'수정-01' 스냅숏 1 (float32)."""
    a = smooth_base(orig, k).astype(np.float32)
    rng = np.random.default_rng(SEED)
    # 노이즈는 화소 단위라서 미리보기에서는 줄여서 원본을 줄였을 때와 비슷하게 보이게 한다
    amp = 255.0 * NOISE * min(1.0, scale)
    noise = rng.uniform(-amp, amp, orig.shape).astype(np.float32)
    b = a + cv2.blur(noise, (3, 3))            # '더 흐리게' ≈ 3×3 흐림
    o = orig.astype(np.float32)
    return LAYER_OPACITY * b + (1 - LAYER_OPACITY) * (LAYER_OPACITY * a + (1 - LAYER_OPACITY) * o)


class SkinStep(Step):
    id = "skin"
    title = "피부 정리"
    param_keys = ("strength.skin",)
    cache = True

    def run(self, frame: Frame, analysis, params) -> Frame:
        st = float(params.get("strength.skin", 1.0))
        regs = frame.regions
        if st <= 0 or "얼굴윤곽" not in regs:
            if "얼굴윤곽" not in regs:
                frame.notes.append("피부: 얼굴 윤곽을 찾지 못해 건너뜀")
            return frame
        k = R.face_k(frame.points)
        x0, y0, x1, y1 = R.face_box(regs, frame.image.shape, 120 * k)
        crop = frame.image[y0:y1, x0:x1]
        off = (x0, y0)
        sm = None if frame.skin_mask is None else frame.skin_mask[y0:y1, x0:x1]

        area = skin_area(crop.shape, regs, sm, off, k)
        # 페더가 눈·코·입 쪽으로 번지지 않게 먼저 페더만큼 줄인 뒤 흐린다
        sigma = FEATHER * k / 2
        alpha = R.feather(R.erode(area, sigma), sigma)

        kept = regs.get("보존점")
        radii = (analysis.extra.get("보존점_r", []) if analysis is not None else [])
        n_kept = 0
        if kept is not None and len(kept) and len(radii) == len(kept):
            prot = np.zeros(crop.shape[:2], np.uint8)
            for (x, y), r in zip(kept, radii):
                rr = r * frame.scale * 1.8 + 2
                cv2.circle(prot, (round(x - x0), round(y - y0)), max(1, round(rr)), 255, -1, cv2.LINE_AA)
                n_kept += 1
            alpha *= 1.0 - R.feather(prot, max(1.0, 6 * k))

        alpha *= min(1.0, BRUSH_OPACITY * st)
        snap = snapshot1(crop, k, frame.scale)
        o = crop.astype(np.float32)
        out = o + alpha[..., None] * (snap - o)
        img = frame.image.copy()
        img[y0:y1, x0:x1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        frame.image = img

        spots = analysis.extra.get("spots", []) if analysis is not None else []
        n_trouble = sum(1 for s in spots if not s.keep)
        if n_kept:
            where = ", ".join(f"({s.x:.0f},{s.y:.0f})" for s in spots if s.keep)
            frame.notes.append(f"피부: 점 {n_kept}개 남김 {where}")
        frame.outputs["skin"] = {"kept": n_kept, "troubles": n_trouble, "k": round(k, 3)}
        return frame
