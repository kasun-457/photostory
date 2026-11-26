"""8. 얼굴 대칭과 윤곽 (가이드 §9, 유동화 + 부분 이동).

가이드 순서대로, 각 항목마다 지금 위치를 다시 재서 움직인다. 변형은 모아 두었다가 한 번만 샘플링한다.
  1. 턱 좌우 (유동화): 입 높이 얼굴 윤곽(초록/E)·턱목 경계(자주/D)에서 넓은 쪽을 좁은 쪽에 맞춘다.
     사진에서 다시 찾은 선(basis '영상')만 쓴다. 추정한 선으로는 고치지 않는다 (보수적).
  2. 코 중심: 코 밑이 중심선에서 5px(원본 기준) 이상 어긋나면 코 조각을 옮긴다.
  3. 입 중심·입꼬리 수평: 입 중심이 5px 이상 어긋나면 옮기고, 입꼬리가 기울면 입 조각을 돌린다.
  4. 눈 높이 (큰 눈 기준): 작은 눈의 눈동자 위·아래 가운데를 큰 눈에 맞춘다.
     눈 '크기' 키우기는 하지 않는다 (본인처럼 보이지 않을 위험, 가이드 §0).
  5. 눈동자 좌우: 홍채 안쪽(주황/C)이 중심선 기준 대칭이 되게 두 눈을 옮긴다 (차이를 반씩).

- 강도: 턱='얼굴형·턱', 코='코', 입='입', 눈='눈'. 강하게 올려도 완전 대칭을 넘지 않는다 (설계서 2-3).
- 끝나면 다시 재서 전과 후 값을 outputs["symmetry"]에 남긴다 (설계서 3-3 비대칭 검사).
- 기준 px 값(5px 등)은 원본 4472px, 얼굴 폭 약 1200px 기준이라 얼굴 폭 배율로 바꾼다.
"""
from __future__ import annotations

import math

import numpy as np

from ..core.measure import regions as R
from ..core.measure.template import eye_opening, measure_template
from ..core.pipeline import Frame, Step
from ..core.warp import Warp

NOSE_MOUTH_MIN = 5.0      # 원본 px (가이드 §9-2)
EYE_MIN = 2.0             # 원본 px (가이드 예: 3px 옮김)
JAW_MIN = 0.008           # 얼굴 폭 대비 (가이드에 숫자 없음, 약 10px)
MOUTH_TILT_MIN = 0.3      # 도 (가이드 예: -0.5°)


def _cap(st: float) -> float:
    return max(0.0, min(1.0, st))


class _Cur:
    """동작마다 지금 위치(점·영역)를 따라간다."""

    def __init__(self, frame: Frame, warp: Warp):
        self.p = dict(frame.points)
        self.r = {k: v.copy() for k, v in frame.regions.items()}
        self.w = warp

    def after_op(self) -> None:
        op = [self.w.ops[-1]]
        names = list(self.p)
        moved = self.w.forward_points(np.array([self.p[k] for k in names]), op)
        self.p = {k: (float(x), float(y)) for k, (x, y) in zip(names, moved)}
        self.r = {k: self.w.forward_points(v, op) if len(v) else v for k, v in self.r.items()}


def _piece(cur: _Cur, keys, grow: float, feather: float, line_key=None, line_w=0.0) -> np.ndarray:
    w = cur.w
    m = np.zeros(w.roi_shape, np.uint8)
    for k in keys:
        if k in cur.r:
            m |= R.poly_mask(w.roi_shape, cur.r[k], hull=True, offset=w.offset())
    if line_key and line_key in cur.r:
        m |= R.line_mask(w.roi_shape, cur.r[line_key], line_w, w.offset())
    # 조각 안은 그대로 옮기고(무게 1), 페더 구간에서만 늘어나도록: 넓힌 뒤 페더만큼 흐림
    return R.feather(R.dilate(m, grow + feather), feather)


class SymmetryStep(Step):
    id = "symmetry"
    title = "얼굴 대칭"
    param_keys = ("strength.jaw", "strength.nose", "strength.mouth", "strength.eyes", "gender")
    cache = True

    def run(self, frame: Frame, analysis, params) -> Frame:
        if "얼굴윤곽" not in frame.regions:
            return frame
        gender = params.get("gender", "여")
        st = {k: _cap(float(params.get(f"strength.{k}", 1.0))) for k in ("jaw", "nose", "mouth", "eyes")}
        p0 = frame.points
        k = R.face_k(p0)
        fw = R.face_w(p0)
        before = measure_template(frame, gender)
        cx = before.center_x
        roi = R.face_box(frame.regions, frame.image.shape, 0.45 * fw)
        warp = Warp(frame.image.shape, roi)
        cur = _Cur(frame, warp)
        done: list[str] = []
        to_orig = 1.0 / frame.scale

        # 1. 턱 좌우 (유동화): 한쪽씩 턱선을 따라 행마다 미는 양을 정한다.
        #    가이드처럼 적용 → 다시 재기 → 남은 만큼 한 번 더 (재기는 마스크만 옮겨서, 샘플링 없이)
        if st["jaw"] > 0:
            t = before
            for rnd in range(2):
                moved_any = False
                for side in ("왼", "오른"):
                    wt, amount, label = self._jaw_push(t, cur, side, cx, fw, gender)
                    if wt is None or amount < JAW_MIN * fw * (0.5 if rnd else 1.0):
                        continue
                    move = amount * st["jaw"] * (1 if side == "왼" else -1)    # 안쪽으로
                    warp.shift(wt, move, 0.0)
                    cur.after_op()
                    moved_any = True
                    if rnd == 0:
                        done.append(f"턱 {side}쪽 {label}")
                if not moved_any or st["jaw"] < 1.0:
                    break
                t = measure_template(self._preview_frame(frame, warp, cur), gender)

        # 2·3. 코·입 중심: 가이드처럼 코와 입을 한 조각으로 넉넉히 오려 함께 옮긴다.
        #      따로 옮기면 두 조각의 페더가 겹쳐 코가 두 번 움직인다 (넘침).
        if (st["nose"] > 0 or st["mouth"] > 0) and "코밑" in cur.p and "왼입꼬리" in cur.p:
            lm, rm = cur.p["왼입꼬리"], cur.p["오른입꼬리"]
            dn = cur.p["코밑"][0] - cx
            dm = (lm[0] + rm[0]) / 2 - cx
            if max(abs(dn), abs(dm)) >= NOSE_MOUTH_MIN * k:
                # 코·입 강도를 각자 반영한 평균 이동
                move = -(dn * st["nose"] + dm * st["mouth"]) / 2
                wt = _piece(cur, ("코아래", "입술"), 0.04 * fw, 0.06 * fw, "콧대", 0.10 * fw)
                warp.shift(wt, move, 0.0)
                cur.after_op()
                done.append(f"코·입 {move * to_orig:+.0f}px")

        # 입꼬리 수평
        if st["mouth"] > 0 and "왼입꼬리" in cur.p:
            lm, rm = cur.p["왼입꼬리"], cur.p["오른입꼬리"]
            tilt = math.degrees(math.atan2(rm[1] - lm[1], rm[0] - lm[0]))
            if abs(tilt) >= MOUTH_TILT_MIN:
                mc = ((lm[0] + rm[0]) / 2, (lm[1] + rm[1]) / 2)
                wt = _piece(cur, ("입술",), 0.04 * fw, 0.06 * fw)
                warp.rotate(wt, mc, -tilt * st["mouth"])
                cur.after_op()
                done.append(f"입꼬리 {-tilt * st['mouth']:+.1f}°")

        # 4. 눈 높이 (큰 눈 기준)
        if st["eyes"] > 0:
            op = {e: eye_opening(cur.p, e) for e in ("왼눈", "오른눈")}
            big = max(op, key=lambda e: op[e][1] - op[e][0])
            small = "오른눈" if big == "왼눈" else "왼눈"
            dy = (op[big][0] + op[big][1]) / 2 - (op[small][0] + op[small][1]) / 2
            if abs(dy) >= EYE_MIN * k:
                wt = _piece(cur, (small,), 0.06 * fw, 0.05 * fw)
                warp.shift(wt, 0.0, dy * st["eyes"])
                cur.after_op()
                done.append(f"{small} 높이 {dy * to_orig:+.0f}px")

        # 5. 눈동자 좌우 (홍채 안쪽 대칭)
        if st["eyes"] > 0 and "왼눈_홍채안쪽" in cur.p and "오른눈_홍채안쪽" in cur.p:
            l, r = cur.p["왼눈_홍채안쪽"][0], cur.p["오른눈_홍채안쪽"][0]
            dl, dr = cx - l, r - cx
            if abs(dl - dr) >= EYE_MIN * k:
                target = (dl + dr) / 2
                for e, sx in (("왼눈", (cx - target) - l), ("오른눈", (cx + target) - r)):
                    wt = _piece(cur, (e,), 0.06 * fw, 0.05 * fw)
                    warp.shift(wt, sx * st["eyes"], 0.0)
                    cur.after_op()
                done.append(f"눈동자 좌우 {abs(dl - dr) * to_orig:.0f}px")

        if not warp.moved:
            frame.outputs["symmetry"] = {"moved": [], "before": _asym(before, frame.points, gender, cx)}
            return frame
        warp.apply_frame(frame)
        after = measure_template(frame, gender)
        # 가이드처럼 안내선(중심선)은 처음 그은 자리에 그대로 두고 잰다
        frame.outputs["symmetry"] = {"moved": done, "before": _asym(before, p0, gender, cx),
                                     "after": _asym(after, frame.points, gender, cx)}
        frame.notes.append("대칭: " + ", ".join(done))
        return frame

    @staticmethod
    def _preview_frame(frame: Frame, warp: Warp, cur: _Cur) -> Frame:
        """다시 재기용: 마스크·점·영역만 옮긴 사본 (이미지는 그대로 — 윤곽은 마스크로 잰다)."""
        f = frame.copy()
        f.person_mask = warp.apply_mask(frame.person_mask)
        f.hair_mask = warp.apply_mask(frame.hair_mask)
        f.skin_mask = warp.apply_mask(frame.skin_mask)
        f.masks = {k: warp.apply_mask(v) for k, v in frame.masks.items()}
        f.points, f.regions = dict(cur.p), dict(cur.r)
        return f

    @staticmethod
    def _jaw_push(t, cur: _Cur, side: str, cx: float, fw: float, gender: str):
        """side 쪽 턱선을 안쪽으로 밀 무게 지도와 최대 이동량.

        입 높이(초록/E)와 턱목 꺾임(자주/D)에서 이쪽이 반대쪽보다 넓은 만큼(넓은 쪽만) 민다.
        두 높이 사이는 선형으로 잇고, 코 밑 위와 꺾임점 아래에서는 0으로 줄인다.
        꺾임점을 사진에서 찾지 못했으면(추정) 그 높이에서는 0 — 모르는 곳은 건드리지 않는다.
        """
        ids = ("초록", "자주", "H", "I") if gender == "여" else ("E", "D", "9", "10")
        contour, bend, hy, iy = (t.get(i) for i in ids)
        if contour is None or contour.basis != "영상" or not contour.sides or hy is None:
            return None, 0.0, ""

        def excess(line):
            if line is None or line.basis != "영상" or not line.sides:
                return 0.0
            l, r = line.sides
            dl, dr = cx - l, r - cx
            return max(0.0, (dl - dr) if side == "왼" else (dr - dl))

        e_m, e_b = excess(contour), excess(bend)
        if max(e_m, e_b) <= 0:
            return None, 0.0, ""
        y_m = hy.pos
        y_b = iy.pos if iy is not None else y_m + 0.25 * fw
        y_top = cur.p["코밑"][1] if "코밑" in cur.p else y_m - 0.2 * fw
        y_end = y_b + 0.10 * fw
        peak = max(e_m, e_b)
        h, w = cur.w.roi_shape
        ox, oy = cur.w.offset()
        ys = np.arange(h, dtype=np.float32) + oy
        prof = np.interp(ys, [y_top, y_m, y_b, y_end], [0.0, e_m, e_b, 0.0], left=0.0, right=0.0) / peak
        # 윤곽 x: 모델 턱선을 입 높이의 실제 윤곽에 맞춰 옮긴 것
        jaw = cur.r.get(f"{side}턱선_선")
        x_act = contour.sides[0] if side == "왼" else contour.sides[1]
        if jaw is not None and len(jaw) >= 2:
            order = np.argsort(jaw[:, 1])
            jy, jx = jaw[order, 1], jaw[order, 0]
            xs = np.interp(ys, jy, jx)
            xs += x_act - float(np.interp(y_m, jy, jx))
        else:
            xs = np.full_like(ys, x_act)
        gx = np.arange(w, dtype=np.float32)[None, :] + ox     # 윤곽 위가 가장 많이(=부족분만큼) 움직임
        wt = np.exp(-0.5 * ((gx - xs[:, None]) / (0.07 * fw)) ** 2) * prof[:, None]
        wt = R.feather(wt.astype(np.float32), 0.02 * fw)
        label = f"입 높이 {e_m:.1f} / 꺾임 {e_b:.1f}px"
        return wt, peak, label


def _asym(t, p, gender, cx: float) -> dict[str, float]:
    """비대칭 값 (px, 지금 해상도). 0에 가까울수록 대칭 (설계서 3-3). cx: 고정한 중심선."""
    out = {}
    for lid, name in ((("초록", "E")[gender != "여"], "볼라인"), (("자주", "D")[gender != "여"], "턱선")):
        line = t.get(lid)
        if line is not None and line.sides:
            out[name] = round((cx - line.sides[0]) - (line.sides[1] - cx), 2)
    if "코밑" in p:
        out["코"] = round(p["코밑"][0] - cx, 2)
    if "왼입꼬리" in p:
        out["입"] = round((p["왼입꼬리"][0] + p["오른입꼬리"][0]) / 2 - cx, 2)
    if "왼눈_홍채안쪽" in p:
        out["눈동자"] = round((cx - p["왼눈_홍채안쪽"][0]) - (p["오른눈_홍채안쪽"][0] - cx), 2)
    return out
