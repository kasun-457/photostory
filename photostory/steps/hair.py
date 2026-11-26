"""머리카락 (가이드 §11·§13).

11. 잔머리 정리 (§11-3): 머리 윤곽 밖으로 삐친 가는 머리카락을 배경으로 돌린다.
    가이드는 생성형 채우기·유동화로 하지만, 배경 위로 삐친 가는 올은 '머리 윤곽을 매끈하게 다듬은 선'
    밖에 있는 가는 부분이라서 사람 영역(마스크)에서 빼면 13번 배경 단계가 배경색으로 칠한다.
    - 머리 윤곽 = 사람 영역 중 턱보다 위, 얼굴 윤곽 밖. 열림 연산(반지름 = 얼굴 폭 1.2% × 강도)으로
      그보다 가는 돌출(올)을 찾는다. 이마·얼굴 위로 내려온 잔머리는 건드리지 않는다 (배경 위만).
    - 핀·머리끈 지우기(§11-1), 머리·얼굴 경계 정리(§11-2)는 물체 찾기와 채우기가 필요해 아직 없다.
14. 머리카락 밝기 (§13): 머리카락이 너무 밝을 때만 머리 부분만 어둡게 (번 도구처럼 밝은 곳을 더).
    판단이 애매하면 하지 않는다. 수정본에서 한다 (전송본에서 고치지 않음).
강도 부위는 둘 다 '머리카락'. 기준값은 공개 사진·합성으로 잡은 값이다 (확실하지 않음).
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core.measure import regions as R
from ..core.pipeline import Frame, Step

STRAND_R = 0.012       # 얼굴 폭 대비 열림 반지름 (이보다 가는 돌출 = 잔머리)
BRIGHT_MEDIAN = 185.0  # 머리카락 L(0~255) 중간값이 이보다 크면 너무 밝음
BLOWN_L, BLOWN_FRAC = 235.0, 0.12   # 또는 거의 흰 화소가 12% 넘으면
TARGET_MEDIAN = 165.0  # 어둡게 한 뒤 중간값 목표


class HairTidyStep(Step):
    id = "hair_tidy"
    title = "잔머리 정리"
    param_keys = ("strength.hair",)
    cache = True

    def run(self, frame: Frame, analysis, params) -> Frame:
        st = max(0.0, min(1.0, float(params.get("strength.hair", 1.0))))
        m = frame.person_mask
        if st <= 0 or m is None or "얼굴윤곽" not in frame.regions or "guides" not in frame.outputs:
            return frame
        fw = R.face_w(frame.points)
        chin = frame.outputs["guides"]["chin_y"]
        crown = frame.outputs["guides"]["crown_y"]
        h, w = m.shape[:2]
        # 머리 둘레만 본다: 정수리 위 여유 ~ 턱 높이, 중심에서 얼굴 폭 1.6배 안
        cx = frame.outputs["guides"]["center_x"]
        x0, x1 = int(max(0, cx - 1.6 * fw)), int(min(w, cx + 1.6 * fw))
        y0, y1 = int(max(0, crown - 0.3 * fw)), int(min(h, chin))
        if x1 <= x0 or y1 <= y0:
            return frame
        sub = m[y0:y1, x0:x1]
        binm = (sub > 0.5).astype(np.uint8) * 255
        r = max(1.0, STRAND_R * fw * st)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * round(r) + 1, 2 * round(r) + 1))
        opened = cv2.morphologyEx(binm, cv2.MORPH_OPEN, k)
        # 얼굴 위(이마로 내려온 머리 포함)는 그대로: 얼굴 윤곽을 넉넉히 넓힌 곳은 지키기
        face = R.dilate(R.poly_mask(sub.shape, frame.regions["얼굴윤곽"], offset=(x0, y0)), 0.06 * fw)
        keep_bin = cv2.bitwise_or(opened, face)
        # 본래 윤곽의 부드러운 경계는 살리고, 윤곽에서 떨어진 올만 0으로
        keep = R.feather(R.dilate(keep_bin, max(1.0, 0.3 * r)), max(0.6, 0.3 * r))
        removed = (binm > 0) & (keep < 0.5)
        if removed.sum() < 0.0005 * fw * fw:
            return frame
        new = sub * np.clip(keep, 0, 1)
        pm = m.copy()
        pm[y0:y1, x0:x1] = new
        frame.person_mask = pm
        if frame.hair_mask is not None:
            hm = frame.hair_mask.copy()
            hm[y0:y1, x0:x1] = frame.hair_mask[y0:y1, x0:x1] * np.clip(keep, 0, 1)
            frame.hair_mask = hm
        frame.outputs["hair_tidy"] = {"removed_px": int(removed.sum()), "radius": round(r, 2)}
        frame.notes.append(f"잔머리: 머리 윤곽 밖 가는 올 정리 ({int(removed.sum() / frame.scale ** 2)}화소)")
        return frame


def hair_core(frame: Frame) -> np.ndarray | None:
    """확실한 머리카락 화소 (얼굴 피부 제외, 사람 영역 안)."""
    hm = frame.hair_mask
    if hm is None:
        return None
    core = hm > 0.7
    if frame.person_mask is not None:
        core &= frame.person_mask > 0.7
    if frame.skin_mask is not None:
        core &= frame.skin_mask < 0.3
    return core


class HairBrightnessStep(Step):
    id = "hair_bright"
    title = "머리카락 밝기"
    param_keys = ("strength.hair",)
    cache = True

    def run(self, frame: Frame, analysis, params) -> Frame:
        st = max(0.0, min(1.0, float(params.get("strength.hair", 1.0))))
        core = hair_core(frame)
        if st <= 0 or core is None or core.sum() < 200:
            return frame
        L = cv2.cvtColor(frame.image, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
        vals = L[core]
        med = float(np.median(vals))
        blown = float((vals > BLOWN_L).mean())
        info = {"median": round(med, 1), "blown": round(blown, 3)}
        if med <= BRIGHT_MEDIAN and blown <= BLOWN_FRAC:
            frame.outputs["hair_bright"] = info | {"applied": False}
            return frame
        # 번 도구(하이라이트): 밝을수록 더 어둡게. L' = L·(1 − a·(L/255)²), 중간값이 목표에 오게 a를 정함
        t = med / 255.0
        a = float(np.clip((1 - TARGET_MEDIAN / max(med, 1.0)) / max(t * t, 1e-3), 0.0, 0.5)) * st
        if blown > BLOWN_FRAC and med <= BRIGHT_MEDIAN:   # 중간값은 괜찮지만 하얗게 날아간 곳이 많을 때
            a = 0.25 * st
        hm = cv2.GaussianBlur(np.where(core, frame.hair_mask, 0).astype(np.float32), (0, 0),
                              max(0.6, 0.004 * R.face_w(frame.points)))
        gain = 1.0 - a * (L / 255.0) ** 2                 # 밝기 배율 (색은 그대로 비율로)
        gain = 1.0 + hm * (gain - 1.0)
        out = frame.image.astype(np.float32) * gain[..., None]
        frame.image = np.clip(out + 0.5, 0, 255).astype(np.uint8)
        after = float(np.median(cv2.cvtColor(frame.image, cv2.COLOR_RGB2LAB)[..., 0][core]))
        frame.outputs["hair_bright"] = info | {"applied": True, "after": round(after, 1), "amount": round(a, 3)}
        frame.notes.append(f"머리카락 밝기: 너무 밝아 어둡게 (중간값 {med:.0f} → {after:.0f})")
        return frame
