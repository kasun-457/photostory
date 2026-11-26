"""5. 눈 반사점 (v1.52 '반대쪽 눈 따오기', 설계서 3-2의 5번).

반사로 뚫려 보이는 눈의 홍채를 반대쪽 눈의 홍채로 덮는다.
- 좌우 반전하지 않는다 → 반사점(빛) 방향이 그대로 유지된다.
- 크기는 두 홍채 반지름 비율로 맞추고, 위치는 반사가 없는 화소끼리 비교해 작게 다시 맞춘다.
- 밝기는 반사 없는 홍채 화소의 채널별 밝기 비율 중앙값으로 맞춘다.
- 홍채 원 ∩ 눈 영역(눈꺼풀 안쪽)만 바꾼다. 가장자리는 부드럽게.
- 판정은 prepare에서 한 번 한다 (core/measure/catchlight.py). 강도 부위는 '눈'.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core.measure import regions as R
from ..core.measure.catchlight import EYES, iris_circle, iris_disk
from ..core.pipeline import Frame, Step


class CatchlightStep(Step):
    id = "catchlight"
    title = "눈 반사점"
    param_keys = ("strength.eyes",)
    cache = True

    def run(self, frame: Frame, analysis, params) -> Frame:
        st = float(params.get("strength.eyes", 1.0))
        info = analysis.extra.get("catchlight", {}) if analysis is not None else {}
        bad = info.get("fix")
        if st <= 0 or bad is None:
            return frame
        good = EYES[1 - EYES.index(bad)]
        p = frame.points
        cb, cg = iris_circle(p, bad), iris_circle(p, good)
        if cb is None or cg is None:
            return frame
        (bc, br), (gc, gr) = cb, cg
        scale = br / max(gr, 1e-3)
        half = int(np.ceil(1.35 * br)) + 2
        img = frame.image
        h, w = img.shape[:2]
        # 고칠 눈은 정수 격자 조각 그대로 쓰고, 반대쪽 눈만 한 번(Lanczos) 샘플링한다
        x0, y0 = int(round(bc[0])) - half, int(round(bc[1])) - half
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + 2 * half + 1), min(h, y0 + 2 * half + 1)
        target = img[ya:yb, xa:xb].astype(np.float32)
        ph, pw = target.shape[:2]
        lc = (bc[0] - xa, bc[1] - ya)                           # 조각 안의 홍채 중심 (소수)

        def donor_at(dx, dy):
            # 조각 화소 X → 반대쪽 눈 gc + (X - lc) / scale (+ 미세 이동)
            M = np.float32([[1 / scale, 0, gc[0] + dx - lc[0] / scale],
                            [0, 1 / scale, gc[1] + dy - lc[1] / scale]])
            return cv2.warpAffine(img, M, (pw, ph), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REFLECT).astype(np.float32)

        disk = iris_disk((ph, pw), lc, 0.92 * br) > 0
        if disk.sum() < 10:
            return frame
        Lt = cv2.cvtColor(np.clip(target, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
        clean_t = disk & (Lt < np.percentile(Lt[disk], 70))            # 반사 없는 화소만 비교
        best, best_d = None, (0.0, 0.0)
        rng = max(1, round(0.15 * gr))
        for dy in range(-rng, rng + 1):
            for dx in range(-rng, rng + 1):
                d = donor_at(dx, dy)
                e = float(np.abs(d[clean_t] - target[clean_t]).mean()) if clean_t.any() else 0.0
                if best is None or e < best:
                    best, best_d = e, (dx, dy)
        donor = donor_at(*best_d)
        Ld = cv2.cvtColor(np.clip(donor, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
        both = clean_t & (Ld < np.percentile(Ld[disk], 70))
        if both.sum() > 10:
            ratio = np.median(target[both] / np.maximum(donor[both], 1.0), axis=0)
            donor = donor * np.clip(ratio, 0.6, 1.6)[None, None, :]
        # 바꿀 영역: 홍채 원 ∩ 눈꺼풀 안쪽, 부드러운 가장자리
        m = iris_disk((ph, pw), lc, 1.02 * br)
        if bad in frame.regions:
            m &= R.poly_mask((ph, pw), frame.regions[bad], offset=(xa, ya))
        alpha = R.feather(m, max(0.6, 0.08 * br)) * min(1.0, st)
        out = target + alpha[..., None] * (donor - target)
        res = img.copy()
        res[ya:yb, xa:xb] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        frame.image = res
        frame.notes.append(f"눈 반사점: {bad} 홍채를 {good}에서 옮겨 덮음 (반사 {info['glare'][bad]:.0%})")
        return frame
