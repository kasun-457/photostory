"""7-1. 콧볼 옆 볼 그림자 지우기 (가이드 §8-7, 설계서 2-3 '명암'에 포함).

사람은 콧볼 옆 볼에 그림자가 지면 바로 옆 깨끗한 피부에서 복제 도장으로 여러 번 덮는다.
여기서는 콧볼 바로 옆 작은 타원 안에서, 둘레 피부보다 뚜렷이 어두운 만큼만 밝혀 둘레 피부에 맞춘다.
- 밝기(큰 흐림)만 바꾸고 피부 결(작은 무늬)은 그대로 둔다 → 도장으로 덮은 것처럼 결이 남는다.
- 코(콧볼·콧구멍)와 입은 건드리지 않는다. 입꼬리 쪽 팔자 주름까지 내려가지 않는다 (콧볼 높이만).
- 둘레보다 덜 어두우면(그림자가 없으면) 하지 않는다.
- 밝기(Lab L)만 올리고 색은 그대로 (RGB를 함께 곱하면 주황빛으로 뜸).
강도 부위는 '명암'. 기준값은 합성으로 잡은 값이다 (확실하지 않음).
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core.measure import regions as R
from ..core.pipeline import Frame, Step

MIN_DEFICIT = 6.0      # Lab L. 그림자 안 어두운 쪽 30% 평균이 둘레보다 이만큼 어두워야 함
MAX_LIFT = 10.0        # 한 화소를 최대 이만큼만 밝힘. 웃을 때 생기는 깊은 팔자 주름은 그림자가 아니라
                       # 얼굴 모양이라 지우면 본인처럼 안 보인다 → 그림자는 지우고 주름은 조금만 옅게


class NoseShadowStep(Step):
    id = "nose_shadow"
    title = "콧볼 옆 그림자"
    param_keys = ("strength.tone",)
    cache = True

    def run(self, frame: Frame, analysis, params) -> Frame:
        st = max(0.0, min(1.0, float(params.get("strength.tone", 1.0))))
        p, regs = frame.points, frame.regions
        if st <= 0 or "왼콧볼" not in p or "오른콧볼" not in p or "코아래" not in regs:
            return frame
        fw = R.face_w(p)
        img = frame.image
        res = img.copy()          # 바꾼 조각만 Lab으로 오가게 (전체를 오가면 모든 화소가 ±1~2 반올림됨)
        h, w = img.shape[:2]
        done = []
        nose_c = (p["왼콧볼"][0] + p["오른콧볼"][0]) / 2
        for side in ("왼", "오른"):
            wx, wy = p[f"{side}콧볼"]
            out_dir = -1.0 if wx < nose_c else 1.0
            cx, cy = wx + out_dir * 0.075 * fw, wy + 0.01 * fw
            ax, ay = 0.055 * fw, 0.065 * fw
            pad = 0.16 * fw
            x0, y0 = int(max(0, cx - pad)), int(max(0, cy - pad))
            x1, y1 = int(min(w, cx + pad)), int(min(h, cy + pad))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crop = img[y0:y1, x0:x1]
            L = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
            Ls = cv2.GaussianBlur(L, (0, 0), max(0.8, 0.015 * fw))          # 밝기(큰 흐림)
            zone = np.zeros(L.shape, np.uint8)
            cv2.ellipse(zone, (round(cx - x0), round(cy - y0)), (max(1, round(ax)), max(1, round(ay))),
                        0, 0, 360, 255, -1)
            ring = np.zeros(L.shape, np.uint8)
            cv2.ellipse(ring, (round(cx - x0), round(cy - y0)), (max(1, round(ax * 1.9)), max(1, round(ay * 1.7))),
                        0, 0, 360, 255, -1)
            ring &= ~R.dilate(zone, 0.01 * fw)
            keep_out = R.dilate(R.poly_mask(L.shape, regs["코아래"], hull=True, offset=(x0, y0)), 0.012 * fw)
            if "입술" in regs:
                keep_out |= R.dilate(R.poly_mask(L.shape, regs["입술"], hull=True, offset=(x0, y0)), 0.03 * fw)
            if frame.skin_mask is not None:
                skin = frame.skin_mask[y0:y1, x0:x1] > 0.5
            else:
                skin = np.ones(L.shape, bool)
            zone_m = (zone > 0) & (keep_out == 0) & skin
            ring_m = (ring > 0) & (keep_out == 0) & skin
            if zone_m.sum() < 20 or ring_m.sum() < 20:
                continue
            ref = float(np.median(Ls[ring_m]))
            deficit = ref - Ls
            dark = np.sort(deficit[zone_m])[::-1]
            depth = float(dark[: max(1, len(dark) * 3 // 10)].mean())
            if depth < MIN_DEFICIT:
                continue
            # 그림자 만큼만 밝힘 (둘레보다 밝은 곳은 그대로), 가장자리는 부드럽게
            soft = R.feather(np.where(zone_m, 255, 0).astype(np.uint8), max(0.8, 0.012 * fw))
            add = np.clip(deficit, 0, MAX_LIFT) * soft * st
            sel = add > 0.25
            lab = cv2.cvtColor(res[y0:y1, x0:x1], cv2.COLOR_RGB2LAB).astype(np.float32)
            lab[..., 0] += add
            new = cv2.cvtColor(np.clip(lab + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
            res[y0:y1, x0:x1][sel] = new[sel]
            done.append(f"{side}쪽 {depth:.0f}")
        if done:
            frame.image = res
            frame.notes.append("콧볼 옆 그림자: " + ", ".join(done) + " 밝힘")
            frame.outputs["nose_shadow"] = done
        return frame
