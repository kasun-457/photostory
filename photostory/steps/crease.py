"""9. 쌍꺼풀 좌우 맞춤 (가이드 §5-3, 설계서 3-2의 9번).

- 쌍꺼풀이 있는 사람만: 윗눈꺼풀과 눈썹 사이에서 주름(어두운 가는 선)을 찾는다.
  눈 가운데 60% 열마다 세로 밝기에서 가장 뚜렷한 골을 찾고, 열들이 매끈한 곡선으로 이어질 때만 주름으로 본다.
- 높이 = 그 눈의 윗눈꺼풀에서 주름의 가장 높은 곳까지 (가이드: 주름의 가장 높은 지점).
- 좌우 높이가 5px(원본 기준) 넘게 다르면 기준 눈에 맞춰 다른 눈의 주름을 옮긴다.
  눈꺼풀과 눈썹은 그대로 두고 그 사이 피부만 늘이고 줄인다 (유동화).
- 기준 눈: 가이드는 좌우가 다르면 사용자에게 한 번 묻는다 (유일한 예외). 처리 중에는 묻지 않으므로
  (설계서 3-2 공통 규칙) 주름이 더 또렷한 눈을 기본으로 하고(가이드 장경미 예: 또렷한 쪽 기준),
  메모로 알린다. Session.crease_ref로 '왼눈'/'오른눈'을 정하면 그것을 쓴다.
- 한쪽만 쌍꺼풀이거나 둘 다 없으면 하지 않는다.
강도 부위는 '눈'. 기준값은 합성으로 잡은 값이다 (확실하지 않음).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..core.measure import regions as R
from ..core.pipeline import Frame, Step
from ..core.warp import Warp

DIFF_MIN = 5.0          # 원본 px (가이드 §5-3: 약 5px 이내면 같다고 봄)
DEPTH_MIN = 4.0         # 회색 밝기. 골이 주변보다 이만큼 어두워야 주름 후보


@dataclass
class Crease:
    x: float            # 가장 높은 곳
    y: float
    lid_y: float        # 그 열의 윗눈꺼풀
    brow_y: float       # 그 열의 눈썹 아래
    height: float       # lid_y − y
    clarity: float      # 골 깊이 중간값
    x0: float
    x1: float
    coef: np.ndarray | None = None      # 주름 곡선 y = 2차식(x)
    cols: np.ndarray | None = None      # 잰 열들의 (x, 눈꺼풀 y, 눈썹 아래 y, 주름 y)


def _col_extent(mask: np.ndarray, x: int, top: bool) -> int | None:
    col = np.nonzero(mask[:, x])[0]
    if col.size == 0:
        return None
    return int(col.min() if top else col.max())


def measure_crease(gray: np.ndarray, regions, eye: str, brow: str) -> Crease | None:
    if eye not in regions or brow not in regions:
        return None
    e, b = regions[eye], regions[brow]
    ex0, ex1 = float(e[:, 0].min()), float(e[:, 0].max())
    ew = ex1 - ex0
    pad = int(0.5 * ew)
    y_top = int(max(0, b[:, 1].min() - 2))
    y_bot = int(min(gray.shape[0], e[:, 1].max() + 2))
    x_lo, x_hi = int(max(0, ex0 - pad)), int(min(gray.shape[1], ex1 + pad))
    sub = gray[y_top:y_bot, x_lo:x_hi]
    off = (x_lo, y_top)
    em = R.poly_mask(sub.shape, e, offset=off) > 0
    bm = R.poly_mask(sub.shape, b, hull=True, offset=off) > 0
    sm = cv2.GaussianBlur(sub, (0, 0), max(0.6, 0.012 * ew))
    pts = []
    for xa in np.linspace(ex0 + 0.1 * ew, ex1 - 0.1 * ew, 16):
        x = int(round(xa)) - x_lo
        lid = _col_extent(em, x, True)
        brow_b = _col_extent(bm, x, False)
        if lid is None or brow_b is None or lid - brow_b < 6:
            continue
        gap = lid - brow_b
        lo, hi = brow_b + int(0.15 * gap), lid - max(2, int(0.12 * gap))   # 눈썹·속눈썹 빼고
        if hi - lo < 3:
            continue
        prof = sm[lo:hi, max(0, x - 1):x + 2].mean(axis=1)
        # 창 안에서 '위아래가 더 밝은 골' 중 가장 뚜렷한 것 (창 끝의 속눈썹·눈썹 어둠은 골이 아님)
        best, k = 0.0, -1
        for j in range(1, len(prof) - 1):
            if prof[j] <= prof[j - 1] and prof[j] <= prof[j + 1]:
                d = min(prof[:j].max(), prof[j:].max()) - prof[j]
                if d > best:
                    best, k = d, j
        depth = best
        if k < 0 or depth < DEPTH_MIN:
            continue
        pts.append((x + x_lo, lo + k + y_top, depth, lid + y_top, brow_b + y_top))
    if len(pts) < 9:
        return None
    a = np.array(pts, np.float64)
    c = np.polyfit(a[:, 0], a[:, 1], 2)
    resid = np.abs(np.polyval(c, a[:, 0]) - a[:, 1])
    if np.median(resid) > 0.03 * ew:                   # 이어지지 않으면 주름이 아님
        return None
    xs = np.linspace(a[:, 0].min(), a[:, 0].max(), 50)
    ys = np.polyval(c, xs)
    i = int(np.argmin(ys))
    lid_y = float(np.interp(xs[i], a[:, 0], a[:, 3]))
    brow_y = float(np.interp(xs[i], a[:, 0], a[:, 4]))
    return Crease(float(xs[i]), float(ys[i]), lid_y, brow_y, lid_y - float(ys[i]), float(np.median(a[:, 2])),
                  ex0, ex1, c, a[:, [0, 3, 4, 1]])


class CreaseStep(Step):
    id = "crease"
    title = "쌍꺼풀"
    param_keys = ("strength.eyes", "crease_ref")
    cache = True

    def run(self, frame: Frame, analysis, params) -> Frame:
        st = max(0.0, min(1.0, float(params.get("strength.eyes", 1.0))))
        if st <= 0:
            return frame
        gray = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        cr = {e: measure_crease(gray, frame.regions, e, b) for e, b in (("왼눈", "왼눈썹"), ("오른눈", "오른눈썹"))}
        to_orig = 1.0 / frame.scale
        info = {e: None if c is None else {"height": round(c.height * to_orig, 1), "clarity": round(c.clarity, 1)}
                for e, c in cr.items()}
        frame.outputs["crease"] = info
        if cr["왼눈"] is None or cr["오른눈"] is None:
            if (cr["왼눈"] is None) != (cr["오른눈"] is None):
                frame.notes.append("쌍꺼풀: 한쪽 눈에서만 찾아 맞추지 않음")
            return frame
        diff = abs(cr["왼눈"].height - cr["오른눈"].height) * to_orig
        if diff < DIFF_MIN:
            return frame
        ref = params.get("crease_ref") or "auto"
        auto = ref not in ("왼눈", "오른눈")
        if auto:
            ref = max(cr, key=lambda e: cr[e].clarity)
        tgt = "오른눈" if ref == "왼눈" else "왼눈"
        c, r = cr[tgt], cr[ref]
        dy = (c.height - r.height) * st                     # +면 주름을 아래로 (높이를 줄임)
        fw = R.face_w(frame.points)
        ew = c.x1 - c.x0
        roi = (c.x0 - 0.5 * ew, c.brow_y - 0.1 * ew, c.x1 + 0.5 * ew, c.lid_y + 0.05 * ew)
        warp = Warp(frame.image.shape, roi, grid=max(1, round(fw / 600)))
        ys, xs = warp.gy[:, 0], warp.gx[0]
        # 열마다 그 열의 눈썹 아래·주름·눈꺼풀 높이로: 눈썹 아래 0 → 주름 1 → 눈꺼풀 0
        # (주름은 곡선이라 한 높이로 잡으면 양 끝이 덜 움직여 꺾인다)
        cx_, lid_, brow_, cre_ = c.cols[:, 0], c.cols[:, 1], c.cols[:, 2], c.cols[:, 3]
        xc = np.clip(xs, cx_.min(), cx_.max())
        lid_x, brow_x = np.interp(xc, cx_, lid_), np.interp(xc, cx_, brow_)
        # 주름은 눈썹 아래와 눈꺼풀 사이에 두고, 눈꺼풀과는 최소 간격을 둔다
        # (주름이 눈꺼풀에 붙으면 무게가 반 화소 만에 1→0이 되어 그 줄이 크게 늘어나 번진다)
        # 열마다 찾은 주름 높이를 그대로 잇는다 (2차 곡선은 눈머리 쪽 급하게 내려가는 끝을 못 따라감)
        cre_x = np.clip(np.interp(xc, cx_, cre_), brow_x + 1.0, lid_x - np.maximum(2.0, 0.2 * (lid_x - brow_x)))
        wt = np.zeros(warp.roi_shape, np.float32)
        for j in range(len(xs)):
            wt[:, j] = np.interp(ys, [brow_x[j], cre_x[j], lid_x[j] - 0.5], [0.0, 1.0, 0.0], left=0.0, right=0.0)
        # 가로: 주름을 잰 범위 안은 그대로 1, 밖으로 눈 폭 15% 동안 부드럽게 0
        dist = np.maximum(0.0, np.maximum(cx_.min() - xs, xs - cx_.max())) / (0.15 * ew)
        horiz = np.where(dist >= 1, 0.0, 0.5 + 0.5 * np.cos(np.pi * np.clip(dist, 0, 1)))
        warp.shift((wt * horiz[None, :]).astype(np.float32), 0.0, dy)
        warp.apply_frame(frame)
        gray2 = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        after = measure_crease(gray2, frame.regions, tgt, "왼눈썹" if tgt == "왼눈" else "오른눈썹")
        info["after_diff"] = None if after is None else round(abs(after.height - r.height) * to_orig, 1)
        info["ref"] = ref
        how = "더 또렷한 눈" if auto else "지정"
        frame.notes.append(f"쌍꺼풀: 좌우 높이 {diff:.0f}px 차이 → {ref} 기준({how})으로 {tgt} 주름을 맞춤. "
                           "기준 눈은 바꿀 수 있음")
        return frame
