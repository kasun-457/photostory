"""10. 목라인 좌우대칭 (가이드 §10).

- 목 좌우 차(오른쪽 폭 − 왼쪽 폭, 원본 px)로 정한다.
  · 20px 미만: 몸을 따지 않고 유동화로만 균형 → 목·몸을 옆으로 부드럽게 민다 (어깨 기울기 그대로).
  · 20px 이상: 몸이 틀어진 것 → 턱목 꺾임(I/10선) 아래 몸 전체를, 눈 높이 중심선 위의 점을 축으로 돌린다.
    θ = atan(목중심_x_중심선차 ÷ (I선_y − 눈_y)). 목 폭은 그대로 둔다 (목을 깎지 않는다).
- 어깨 수평도 함께 본다: 돌린 뒤 어깨가 눈에 띄게 기울면(2° 넘고 전보다 나쁨) 각도를 절반으로,
  그래도 그러면 하지 않는다 (가이드: 목 수치만 보고 크게 돌리면 부자연스럽다).
- 얼굴(턱 포함)은 움직이지 않는다. 얼굴과 목 사이는 부드럽게 이어지는 변형이라 따로 채우기가 필요 없다.
- 목을 사진에서 또렷하게 재지 못하면(옷깃·머리카락에 가림, 행마다 들쭉날쭉) 하지 않는다 (보수적).
- 강도 부위는 '목·어깨'. 1을 넘겨도 완전 대칭에서 멈춤.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.measure import regions as R
from ..core.measure.template import _row_edges, _skin_union
from ..core.pipeline import Frame, Step
from ..core.warp import Warp

BIG = 20.0            # 원본 px (가이드 §10)
SMALL_MIN = 3.0       # 원본 px, 이보다 작으면 하지 않음
SHOULDER_BAD = 2.0    # 도


@dataclass
class Neck:
    offset: float      # 목 중심 x − 중심선 (지금 해상도 px). +면 오른쪽
    asym: float        # 오른쪽 폭 − 왼쪽 폭 (= 2·offset)
    width: float
    rows: int


def measure_neck(union: np.ndarray, cx: float, y0: float, y1: float, fw: float) -> Neck | None:
    """y0~y1 행마다 피부(얼굴+몸) 윤곽을 중심선에서 바깥으로 재서 목 가운데·폭을 구한다."""
    offs, widths = [], []
    for y in np.linspace(y0, y1, 16):
        l, r = _row_edges(union, int(round(y)), cx, cx, cx)
        if l is None or r is None:
            continue
        w = r - l
        if not (0.45 * fw <= w <= 1.2 * fw):           # 목이라기엔 너무 좁거나(가림) 넓음(옷깃·어깨)
            continue
        offs.append((l + r) / 2 - cx)
        widths.append(w)
    if len(offs) < 6:
        return None
    offs = np.array(offs)
    if float(np.std(offs)) > 0.04 * fw:               # 행마다 들쭉날쭉 → 믿지 않음
        return None
    off = float(np.median(offs))
    return Neck(off, 2 * off, float(np.median(widths)), len(offs))


def shoulder_tilt(top_mask: np.ndarray | None, cx: float, y_from: float, fw: float) -> float | None:
    """어깨 기울기 (도, +면 사진 오른쪽이 아래). 옷·몸이 시작되는 높이를 중심선 양쪽 바깥 띠에서 잰다."""
    if top_mask is None:
        return None
    h, w = top_mask.shape[:2]
    ys = []
    for sgn in (-1, 1):
        tops = []
        for x in np.linspace(cx + sgn * 0.75 * fw, cx + sgn * 1.0 * fw, 6):
            xi = int(round(x))
            if not 0 <= xi < w:
                continue
            col = top_mask[int(max(0, y_from)):, xi] > 0.5
            idx = np.flatnonzero(col)
            if idx.size:
                tops.append(y_from + idx[0])
        if len(tops) < 3:
            return None
        ys.append(float(np.median(tops)))
    dx = 1.75 * fw
    return math.degrees(math.atan2(ys[1] - ys[0], dx))


def decide(asym_orig: float, offset: float, lever: float, tilt: float | None, st: float):
    """(방법, 값). 방법: None | "shift"(px, 지금 해상도) | "rotate"(도, 시계 +)."""
    if abs(asym_orig) < SMALL_MIN or st <= 0:
        return None, 0.0
    if abs(asym_orig) < BIG:
        return "shift", -offset * st
    deg = math.degrees(math.atan2(offset, lever)) * st   # 목이 오른쪽(+)이면 시계(+)로 돌려 왼쪽으로
    if tilt is None:
        return "rotate", deg
    for d in (deg, deg / 2):
        after = tilt + d
        if not (abs(after) > SHOULDER_BAD and abs(after) > abs(tilt)):
            return "rotate", d
    return None, 0.0


class NeckStep(Step):
    id = "neck"
    title = "목라인 대칭"
    param_keys = ("strength.neck", "gender")
    cache = True

    def run(self, frame: Frame, analysis, params) -> Frame:
        st = max(0.0, min(1.0, float(params.get("strength.neck", 1.0))))
        t = frame.outputs.get("template")
        if t is None or "얼굴윤곽" not in frame.regions:
            return frame
        p = frame.points
        fw = R.face_w(p)
        cx = t.center_x
        bend = t.get("I") or t.get("10")
        if bend is None:
            return frame
        if bend.basis != "영상":
            frame.notes.append("목: 턱·목 꺾임을 사진에서 찾지 못해 하지 않음")
            return frame
        union = _skin_union(frame)
        neck = None if union is None else measure_neck(union, cx, bend.pos + 0.03 * fw, bend.pos + 0.18 * fw, fw)
        if neck is None:
            frame.notes.append("목: 목 경계를 또렷하게 재지 못해 하지 않음")
            return frame
        top = None
        for m in (frame.masks.get("옷"), frame.masks.get("몸피부")):
            if m is not None:
                top = m if top is None else np.maximum(top, m)
        chin = (t.get("J") or t.get("11")).pos
        tilt = shoulder_tilt(top, cx, chin, fw)
        eye_y = (p["왼눈_바깥꼭짓점"][1] + p["오른눈_바깥꼭짓점"][1]) / 2
        lever = max(1.0, bend.pos - eye_y)
        to_orig = 1.0 / frame.scale
        method, val = decide(neck.asym * to_orig, neck.offset, lever, tilt, st)
        info = {"before": round(neck.asym * to_orig, 1), "tilt_before": None if tilt is None else round(tilt, 2),
                "method": method, "value": round(val, 3)}
        if method is None:
            frame.outputs["neck"] = info
            if abs(neck.asym * to_orig) >= SMALL_MIN:
                frame.notes.append(f"목: 좌우 차 {neck.asym * to_orig:.0f}px, 어깨가 기울어 돌리지 않음")
            return frame

        h, w = frame.image.shape[:2]
        y_top = bend.pos - 0.12 * fw
        warp = Warp(frame.image.shape, (0, y_top, w, h), grid=max(1, round(fw / 150)))
        # 무게: 꺾임선 조금 위에서 0 → 아래로 1, 얼굴(턱 포함)은 0
        ys = warp.gy[:, 0]
        # 가이드는 I선 아래를 통째로 오린다 → I선에서 이미 1. 턱(얼굴)은 아래 얼굴 마스크로 지킨다
        ramp = np.clip((ys - (bend.pos - 0.08 * fw)) / (0.10 * fw), 0, 1).astype(np.float32)
        wt = np.repeat(ramp[:, None], warp.roi_shape[1], axis=1)
        face = R.poly_mask(warp.roi_shape, warp.to_local(frame.regions["얼굴윤곽"]))
        # 턱 둘레 보호는 좁게: 목 옆선은 턱선 바로 아래에서 시작하므로 넓게 지키면 목 위쪽이 덜 움직인다
        face = R.feather(R.dilate(face, warp.px(0.015 * fw)), warp.px(0.025 * fw))
        wt *= 1.0 - face
        if method == "shift":
            warp.shift(wt, val, 0.0)
            frame.notes.append(f"목: 좌우 차 {neck.asym * to_orig:.0f}px → 몸을 {val * to_orig:+.0f}px 밀어 맞춤 (유동화)")
        else:
            warp.rotate(wt, (cx, eye_y), val)
            frame.notes.append(f"목: 좌우 차 {neck.asym * to_orig:.0f}px → 몸 {val:+.2f}° 회전")
        warp.apply_frame(frame)
        after = measure_neck(_skin_union(frame), cx, bend.pos + 0.03 * fw, bend.pos + 0.18 * fw, fw)
        top2 = None
        for m in (frame.masks.get("옷"), frame.masks.get("몸피부")):
            if m is not None:
                top2 = m if top2 is None else np.maximum(top2, m)
        tilt2 = shoulder_tilt(top2, cx, chin, fw)
        info.update(after=None if after is None else round(after.asym * to_orig, 1),
                    tilt_after=None if tilt2 is None else round(tilt2, 2),
                    width_before=round(neck.width * to_orig, 1),
                    width_after=None if after is None else round(after.width * to_orig, 1))
        frame.outputs["neck"] = info
        return frame
