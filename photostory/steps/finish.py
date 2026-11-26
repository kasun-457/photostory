"""8-1. 마무리 유동화 (가이드 §9-5: 눈/코/입 수평, 턱 좌우, 턱선, 눈썹, 머리 윤곽·크기, 얼굴 좌우 밸런스).

8번 대칭 단계가 이미 한 것(턱 좌우·턱선, 코·입 중심, 입꼬리 수평, 눈 높이·눈동자)은 빼고,
남은 세 가지를 한다. 대칭 단계처럼 동작마다 지금 위치를 다시 재고, 변형은 모아 한 번만 샘플링한다.

  1. 눈썹 높이: 두 눈썹의 실제 아래 끝(B/2번 선을 재는 방법)이 다르면 두 눈썹을 반씩 옮겨 맞춘다.
     한쪽만 크게 옮기면 표정이 달라 보일 수 있어서 반씩. 눈꺼풀은 움직이지 않게 무게를 눈썹 아래에서 0으로.
  2. 얼굴 좌우 밸런스 (볼): 콧볼 높이(G/7번)의 실제 얼굴 윤곽이 좌우 다르면 넓은 쪽을 좁은 쪽에 맞춘다
     (가이드 §5 '좁은 쪽'). 눈 높이와 입 높이에서는 0 — 입 높이 아래는 대칭 단계의 턱 맞추기가 맡는다.
     머리카락이 볼에 닿아 있는 쪽은 윤곽이 얼굴인지 머리카락인지 몰라서 하지 않는다.
  3. 머리 윤곽: 정수리~눈 높이에서 머리(머리카락+피부) 바깥 폭이 좌우 다르면 넓은 쪽을 안쪽으로 민다.
     차이가 너무 크면(얼굴 폭 6% 넘게) 가르마·머리 모양이라 보고 하지 않는다.

강도: 눈썹='눈썹', 볼='얼굴형·턱', 머리 윤곽='머리카락'. 1을 넘겨도 완전 대칭에서 멈춘다.
눈 크기 키우기는 여전히 하지 않는다 (본인처럼 안 보일 위험, 대칭 단계 설명 참고).
기준값(3px, 6% 등)은 가이드에 숫자가 없어 정한 값이다 (확실하지 않음 — 실제 사진으로 맞춰야 함).
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core.measure import regions as R
from ..core.measure.basic import center_x
from ..core.measure.template import _edge_down, _row_edges, _skin_union
from ..core.pipeline import Frame, Step
from ..core.warp import Warp

BROW_MIN = 3.0          # 원본 px (얼굴 폭 1200px 기준)
CHEEK_MIN = 0.008       # 얼굴 폭 대비 (대칭 단계 턱과 같음)
HAIR_MIN = 0.012        # 얼굴 폭 대비
HAIR_STYLE = 0.06       # 이보다 크게 다르면 머리 모양으로 본다
HAIR_CAP = 0.04         # 한 행에서 미는 최대량 (얼굴 폭 대비)


def _cap(st: float) -> float:
    return max(0.0, min(1.0, st))


def brow_bottoms(gray: np.ndarray, regions, fw: float) -> dict[str, float] | None:
    """두 눈썹 각자의 실제 아래 끝 y (가운데 40% 띠). 하나라도 사진에서 못 찾으면 None."""
    out = {}
    for key in ("왼눈썹", "오른눈썹"):
        if key not in regions:
            return None
        poly = regions[key]
        x_lo, x_hi = poly[:, 0].min(), poly[:, 0].max()
        y_top, y_bot = poly[:, 1].min(), poly[:, 1].max()
        bh = max(2.0, y_bot - y_top)
        b = _edge_down(gray, x_lo + 0.3 * (x_hi - x_lo), x_lo + 0.7 * (x_hi - x_lo),
                       y_top - 0.3 * bh, y_bot + 0.6 * bh, 0.004 * fw)
        if b is None:
            return None
        out[key] = b
    return out


def head_rows(mask: np.ndarray, cx: float, y0: float, y1: float, step: int):
    """y0~y1 행마다 중심선에서 끊기지 않고 이어진 마스크의 왼쪽·오른쪽 끝 x. (ys, 왼, 오른)"""
    h, w = mask.shape[:2]
    ys, L, Rr = [], [], []
    c = int(round(cx))
    for y in range(int(max(0, y0)), int(min(h, y1)), max(1, step)):
        row = mask[y] > 0.5
        if not row[c]:
            continue
        nz = np.flatnonzero(~row[:c])
        l = (nz[-1] + 1) if nz.size else 0
        nz = np.flatnonzero(~row[c:])
        r = (c + nz[0] - 1) if nz.size else w - 1
        ys.append(y)
        L.append(float(l))
        Rr.append(float(r))
    return np.array(ys, np.float32), np.array(L, np.float32), np.array(Rr, np.float32)


class FinishStep(Step):
    id = "finish"
    title = "마무리 유동화"
    param_keys = ("strength.brows", "strength.jaw", "strength.hair")
    cache = True

    def run(self, frame: Frame, analysis, params) -> Frame:
        if "얼굴윤곽" not in frame.regions:
            return frame
        st = {k: _cap(float(params.get(f"strength.{k}", 1.0))) for k in ("brows", "jaw", "hair")}
        if max(st.values()) <= 0:
            return frame
        p = frame.points
        fw = R.face_w(p)
        k = R.face_k(p)
        g = frame.outputs.get("guides", {})
        cx = float(g.get("center_x", center_x(p)))
        crown = float(g.get("crown_y", R.face_box(frame.regions, frame.image.shape, 0)[1] - 0.3 * fw))
        mouth_y = (p["왼입꼬리"][1] + p["오른입꼬리"][1]) / 2 if "왼입꼬리" in p else p["왼볼_입높이"][1]
        roi = (cx - 1.2 * fw, crown - 0.15 * fw, cx + 1.2 * fw, mouth_y + 0.1 * fw)
        warp = Warp(frame.image.shape, roi, grid=max(1, round(fw / 300)))
        to_orig = 1.0 / frame.scale
        done: list[str] = []
        info: dict = {}

        if st["brows"] > 0:
            r = self._brows(frame, warp, fw, k, st["brows"])
            if r:
                done.append(r[0])
                info["brows"] = r[1]
        if st["jaw"] > 0:
            r = self._cheeks(frame, warp, cx, fw, st["jaw"], to_orig)
            if r:
                done.append(r[0])
                info["cheek"] = r[1]
        if st["hair"] > 0:
            r = self._hair(frame, warp, cx, crown, fw, st["hair"], to_orig)
            if r:
                done.append(r[0])
                info["hair"] = r[1]
        frame.outputs["finish"] = {"moved": done, **info}
        if warp.moved:
            warp.apply_frame(frame)
            frame.notes.append("마무리 유동화: " + ", ".join(done))
        return frame

    # ------------------------------------------------------------ 1. 눈썹 높이
    def _brows(self, frame: Frame, warp: Warp, fw: float, k: float, st: float):
        gray = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        b = brow_bottoms(gray, frame.regions, fw)
        if b is None:
            return None
        diff = b["왼눈썹"] - b["오른눈썹"]                       # +: 왼눈썹이 더 낮음
        if abs(diff) < BROW_MIN * k:
            return None
        ys = warp.gy[:, 0]
        for key, eye, sign in (("왼눈썹", "왼눈", -1.0), ("오른눈썹", "오른눈", 1.0)):
            m = R.poly_mask(warp.roi_shape, warp.to_local(frame.regions[key]), hull=True)
            m = R.feather(R.dilate(m, warp.px(0.05 * fw)), warp.px(0.03 * fw))
            # 눈꺼풀은 그대로: 눈썹 아래 끝에서 윗눈꺼풀 사이에 무게를 0으로 내린다
            lid = float(frame.regions[eye][:, 1].min()) if eye in frame.regions else b[key] + 0.08 * fw
            lo = b[key] + 0.25 * max(1.0, lid - b[key])
            hi = lid - 0.15 * max(1.0, lid - b[key])
            fall = np.clip((hi - ys) / max(1.0, hi - lo), 0.0, 1.0)
            warp.shift((m * fall[:, None]).astype(np.float32), 0.0, sign * diff / 2 * st)
        return (f"눈썹 높이 {abs(diff) * (1.0 / frame.scale):.0f}px (반씩)", {"before": round(diff / frame.scale, 1)})

    # ------------------------------------------------------------ 2. 볼 (얼굴 좌우 밸런스)
    def _cheeks(self, frame: Frame, warp: Warp, cx: float, fw: float, st: float, to_orig: float):
        p = frame.points
        union = _skin_union(frame)
        if union is None or "왼콧볼" not in p or "왼눈_바깥꼭짓점" not in p:
            return None
        union = warp.apply_mask(union) if warp.moved else union
        y_w = int(round(max(p["왼콧볼"][1], p["오른콧볼"][1])))
        sl = p["왼볼_입높이"][0] + 0.15 * (cx - p["왼볼_입높이"][0])
        sr = p["오른볼_입높이"][0] - 0.15 * (p["오른볼_입높이"][0] - cx)
        l, r = _row_edges(union, y_w, sl, sr, cx)
        if l is None or r is None:
            return None
        dl, dr = cx - l, r - cx
        side = "왼" if dl > dr else "오른"
        excess = abs(dl - dr)
        if excess < CHEEK_MIN * fw:
            return None
        hair = frame.hair_mask
        if hair is not None:
            hx = int(round(l - 0.015 * fw)) if side == "왼" else int(round(r + 0.015 * fw))
            hh, ww = hair.shape[:2]
            if 0 <= hx < ww and hair[max(0, y_w - 2):y_w + 3, hx].max() > 0.5:
                return None
        y_eye = (p["왼눈_바깥꼭짓점"][1] + p["오른눈_바깥꼭짓점"][1]) / 2
        y_m = (p["왼입꼬리"][1] + p["오른입꼬리"][1]) / 2 if "왼입꼬리" in p else y_w + 0.15 * fw
        ys, gx = warp.gy[:, 0], warp.gx[0][None, :]
        prof = np.interp(ys, [y_eye, y_w, y_m], [0.0, 1.0, 0.0], left=0.0, right=0.0)
        x_edge = l if side == "왼" else r
        wt = np.exp(-0.5 * ((gx - x_edge) / (0.07 * fw)) ** 2) * prof[:, None]
        move = excess * st * (1 if side == "왼" else -1)
        warp.shift(R.feather(wt.astype(np.float32), warp.px(0.02 * fw)), move, 0.0)
        return f"볼 {side}쪽 {excess * to_orig:.0f}px", {"before": round((dl - dr) * to_orig, 1)}

    # ------------------------------------------------------------ 3. 머리 윤곽
    def _hair(self, frame: Frame, warp: Warp, cx: float, crown: float, fw: float, st: float, to_orig: float):
        p = frame.points
        if frame.hair_mask is None:
            return None
        head = frame.hair_mask
        union = _skin_union(frame)
        if union is not None:
            head = np.maximum(head, union)
        if warp.moved:
            head = warp.apply_mask(head)
        y_eye = (p["왼눈_바깥꼭짓점"][1] + p["오른눈_바깥꼭짓점"][1]) / 2
        y0 = crown + 0.08 * (y_eye - crown)                     # 정수리 바로 아래는 폭이 좁아 뺀다
        ys, L, Rr = head_rows(head, cx, y0, y_eye, max(1, round(fw / 300)))
        if ys.size < 8:
            return None
        dl, dr = cx - L, Rr - cx
        ex = dl - dr                                            # +: 왼쪽이 넓음
        med = float(np.median(ex))
        if abs(med) < HAIR_MIN * fw:
            return None
        if abs(med) > HAIR_STYLE * fw:
            frame.notes.append("마무리 유동화: 머리 좌우 폭 차이가 커서(머리 모양) 머리 윤곽은 그대로 둠")
            return None
        side = "왼" if med > 0 else "오른"
        sgn = 1.0 if side == "왼" else -1.0
        amt = np.clip(sgn * ex, 0.0, HAIR_CAP * fw)
        # 행마다 들쭉날쭉한 곳(삐친 머리)은 부드럽게: 이동량을 세로로 평균
        kk = max(1, ys.size // 8)
        amt = np.convolve(np.pad(amt, kk, mode="edge"), np.ones(2 * kk + 1) / (2 * kk + 1), "valid")
        peak = float(amt.max())
        if peak <= 0:
            return None
        gy = warp.gy[:, 0]
        edge = L if side == "왼" else Rr
        prof = np.interp(gy, ys, amt / peak, left=0.0, right=0.0)
        # 위·아래 끝은 부드럽게 0으로
        span = max(1.0, float(ys[-1] - ys[0]))
        taper = np.clip(np.minimum(gy - ys[0], ys[-1] - gy) / (0.15 * span), 0.0, 1.0)
        xe = np.interp(gy, ys, edge)
        wt = np.exp(-0.5 * ((warp.gx[0][None, :] - xe[:, None]) / (0.05 * fw)) ** 2) * (prof * taper)[:, None]
        # 얼굴 피부는 지킨다 (관자놀이 쪽 얼굴이 따라 움직이지 않게)
        if frame.skin_mask is not None:
            sk = frame.skin_mask
            hh, ww = sk.shape[:2]
            yi = np.clip(np.round(warp.gy[:, 0]).astype(int), 0, hh - 1)
            xi = np.clip(np.round(warp.gx[0]).astype(int), 0, ww - 1)
            wt = wt * (1.0 - np.clip(sk[np.ix_(yi, xi)], 0, 1))
        warp.shift(R.feather(wt.astype(np.float32), warp.px(0.015 * fw)), sgn * peak * st, 0.0)
        return f"머리 윤곽 {side}쪽 {peak * to_orig:.0f}px", {"before": round(med * to_orig, 1)}
