"""15. 어깨 좌우 밸런스 (가이드 §14).

사람은 전송본에 앉힌 뒤 어디서 잘리는지 보고 어깨 끝 위치를 맞췄다. 여기서는 잘리는 위치를
미리 계산해서(transfer.crop_rect, 가이드 §14-1 식) 수정본에서 맞춘다.
- 규격마다 트리밍 사각형의 왼쪽·오른쪽 변에서 어깨(사람 영역 윤곽)가 시작되는 높이를 잰다.
- 높은 쪽 어깨를 유동화로 낮은 쪽에 맞춘다 (가이드: 높은 쪽을 낮은 쪽에).
- 규격이 여러 개면 규격마다(변 위치가 다름) 차례로 맞추고, 끝나면 모두 다시 잰다.
- 전송본 기준 1px 이내면 하지 않는다 (가이드: 좌우 차 1~2px 이내 확인).
- 머리카락이 변에서 어깨를 덮으면 그 규격은 하지 않는다 (머리카락을 끌어내리면 머리결이 휨).
- 차이가 얼굴 폭의 8%를 넘으면 하지 않는다: 몸을 틀었거나 옷 모양 탓이라 유동화로 맞추면 크게 일그러진다.
- 트리밍 밖은 손대지 않아도 되지만, 유동화는 부드럽게 이어지므로 밖도 조금 같이 움직인다.
- 강도 부위는 '목·어깨'. 1을 넘겨도 좌우가 같아지는 데서 멈춤.
"""
from __future__ import annotations

import numpy as np

from ..core.measure import regions as R
from ..core.pipeline import Frame, Step
from ..core.session import Placement
from ..core.warp import Warp, gaussian_weight
from .transfer import crop_rect

TOL_PX = 1.0       # 전송본 px
MAX_DIFF = 0.08    # 얼굴 폭 대비. 이보다 크면 자세·옷(몸을 튼 자세, 옷깃 장식) 문제라 유동화로 맞추지 않음


def shoulder_at(person: np.ndarray, hair: np.ndarray | None, x: float, y_from: float,
                y_to: float) -> tuple[float | None, str]:
    """x 열에서 y_from 아래로 내려가며 사람 영역이 시작되는 y. (y, 이유)"""
    h, w = person.shape[:2]
    xi = int(round(x))
    if not 0 <= xi < w:
        return None, "사진 밖"
    ya, yb = int(max(0, y_from)), int(min(h, y_to))
    if yb <= ya:
        return None, "범위 없음"
    col = person[ya:yb, xi] > 0.5
    if col[0]:
        return None, "변 위쪽부터 사람(머리) 영역"
    idx = np.flatnonzero(col)
    if idx.size == 0:
        return None, "트리밍 안에 어깨가 없음"
    y = float(ya + idx[0])
    if hair is not None and hair[int(min(h - 1, y + 2)), xi] > 0.5:
        return None, "머리카락이 어깨를 덮음"
    return y, ""


class ShoulderStep(Step):
    id = "shoulder"
    title = "어깨 밸런스"
    param_keys = ("strength.neck", "specs", "placements")
    cache = True

    def _measure(self, frame: Frame, person, hair, spec, pl, chin):
        x0, y0, x1, y1, s = crop_rect(frame, spec, pl)
        inset = 1.5 / s                       # 변에서 전송본 1.5px 안쪽 열
        yl, why_l = shoulder_at(person, hair, x0 + inset, max(y0, chin), y1)
        yr, why_r = shoulder_at(person, hair, x1 - inset, max(y0, chin), y1)
        return (x0 + inset, yl, why_l), (x1 - inset, yr, why_r), s

    def run(self, frame: Frame, analysis, params) -> Frame:
        st = max(0.0, min(1.0, float(params.get("strength.neck", 1.0))))
        specs = params.get("specs") or []
        if st <= 0 or not specs or frame.person_mask is None or "guides" not in frame.outputs:
            return frame
        p = frame.points
        fw = R.face_w(p)
        chin = frame.outputs["guides"]["chin_y"]
        placements = params.get("placements") or {}
        h, w = frame.image.shape[:2]
        warp = Warp(frame.image.shape, (0, chin, w, h), grid=max(1, round(fw / 150)))
        person, hair = frame.person_mask, frame.hair_mask
        report, notes = {}, []
        # 변이 중심에 가까운 규격부터 (안쪽을 먼저 맞춰야 바깥 규격이 안쪽 결과를 다시 흔들지 않음)
        order = sorted(specs, key=lambda sp: crop_rect(frame, sp, placements.get(sp.id) or Placement())[2],
                       reverse=True)
        # 규격마다 변 위치가 달라 유동화 범위가 겹친다 → 가이드처럼 '맞추기 → 다시 재기'를 몇 번 되풀이
        for rnd in range(3):
            pushed = False
            for spec in order:
                if "skip" in report.get(spec.id, {}):
                    continue
                pl = placements.get(spec.id) or Placement()
                if warp.moved:
                    person = warp.apply_mask(frame.person_mask)
                    hair = warp.apply_mask(frame.hair_mask)
                (xl, yl, wl), (xr, yr, wr), s = self._measure(frame, person, hair, spec, pl, chin)
                if yl is None or yr is None:
                    if rnd == 0:
                        report[spec.id] = {"skip": wl or wr}
                        notes.append(f"{spec.name}: {wl or wr}")
                    continue
                diff = yl - yr                         # +면 왼쪽이 낮음(오른쪽이 높음)
                if rnd == 0:
                    report[spec.id] = {"before": round(diff * s, 2)}
                    if abs(diff) > MAX_DIFF * fw:
                        report[spec.id]["skip"] = "차이가 너무 큼 (자세·옷)"
                        notes.append(f"{spec.name}: 어깨 차이 {abs(diff) * s:.0f}px(전송본)가 너무 커서 하지 않음")
                        continue
                if abs(diff) * s < TOL_PX * (1.0 if rnd == 0 else 0.5):
                    continue
                high_x, high_y = (xr, yr) if diff > 0 else (xl, yl)
                wt = gaussian_weight(warp, (high_x, high_y), 0.20 * fw, 0.12 * fw)
                warp.shift(wt, 0.0, abs(diff) * st)    # 높은 쪽을 아래로
                pushed = True
                if rnd == 0:
                    notes.append(f"{spec.name}: {'오른' if diff > 0 else '왼'}쪽 어깨 {abs(diff) * s:.1f}px(전송본) 내림")
            if not pushed or st < 1.0:
                break

        if warp.moved:
            warp.apply_frame(frame)
        for spec in order:                         # 모두 다시 잰다
            if "skip" in report.get(spec.id, {}):
                continue
            pl = placements.get(spec.id) or Placement()
            (_, yl, _), (_, yr, _), s = self._measure(frame, frame.person_mask, frame.hair_mask, spec, pl, chin)
            if yl is not None and yr is not None:
                report[spec.id]["after"] = round((yl - yr) * s, 2)
        frame.outputs["shoulder"] = report
        if notes:
            frame.notes.append("어깨: " + ", ".join(notes))
        return frame
