"""측정 안내선 (가이드 §5 여자 5-1 / 남자 5-2 템플릿).

수평 맞춘 뒤의 이미지에서 잰다. 결과는 가로선(y)·세로선(x) 목록이다.
- 세로 대칭쌍(얼굴 윤곽·턱목 경계·머리 폭 등): 중심선에서 좌우를 재고 '좁은 쪽' 거리를 양쪽에 쓴다 (§5-0).
- 각 선은 어떻게 쟀는지(basis)를 함께 낸다: "모델"(얼굴 점 그대로), "영상"(사진 경계에서 다시 찾음),
  "추정"(찾지 못해 근사). 2단계 대칭 보정은 '실제 경계 − 안내선'을 보정량으로 쓴다.

아직 없는 것: 귀 끝선(남자 8번), 쌍꺼풀선(§5-3), 눈 윤곽 위 선(§5-5), 얼굴살 사람의 그림자 턱선(§5-4).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import regions as R
from .basic import center_x


@dataclass
class Line:
    id: str            # 가이드 이름 (여: A~K, 주황 등 / 남: 1~12, A~F)
    title: str
    pos: float         # 가로선 y, 세로선은 중심선 기준 거리 (대칭쌍) 또는 x
    basis: str = "모델"
    pair: bool = False  # 세로 대칭쌍이면 True (pos = 중심선에서의 거리)
    sides: tuple[float, float] | None = None   # 대칭쌍: 실제로 잰 왼쪽·오른쪽 x


@dataclass
class Template:
    gender: str
    center_x: float
    h: list[Line] = field(default_factory=list)
    v: list[Line] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def get(self, line_id: str) -> Line | None:
        return next((l for l in self.h + self.v if l.id == line_id), None)

    def v_x(self, line: Line) -> list[float]:
        if line.pair:
            return [self.center_x - line.pos, self.center_x + line.pos]
        return [line.pos]


# ------------------------------------------------------------ 영상에서 경계 찾기

def _edge_down(gray: np.ndarray, x0: float, x1: float, y0: float, y1: float, sigma: float) -> float | None:
    """[x0,x1] 띠의 세로 밝기에서, 가장 어두운 곳 아래로 '어둠 → 밝음' 경계가 가장 센 y."""
    h, w = gray.shape[:2]
    xa, xb = int(max(0, x0)), int(min(w, x1 + 1))
    ya, yb = int(max(0, y0)), int(min(h, y1 + 1))
    if xb - xa < 1 or yb - ya < 5:
        return None
    prof = gray[ya:yb, xa:xb].mean(axis=1).astype(np.float32)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 0), max(0.6, sigma)).ravel()
    k0 = int(np.argmin(prof))
    g = np.gradient(prof)[k0:]
    if g.size < 2 or g.max() <= 0.5:
        return None
    return float(ya + k0 + int(np.argmax(g)))


def _brow_lines(gray, regs, fw) -> tuple[float, float, str] | None:
    """(눈썹 위, 실제 눈썹 아래 끝, basis). 처진 꼬리를 빼려고 눈썹 가운데 40%만 본다."""
    tops, bottoms, basis = [], [], "영상"
    for key in ("왼눈썹", "오른눈썹"):
        if key not in regs:
            return None
        poly = regs[key]
        x_lo, x_hi = poly[:, 0].min(), poly[:, 0].max()
        cx0, cx1 = x_lo + 0.3 * (x_hi - x_lo), x_lo + 0.7 * (x_hi - x_lo)
        y_top, y_bot = poly[:, 1].min(), poly[:, 1].max()
        bh = max(2.0, y_bot - y_top)
        tops.append(float(y_top))
        b = _edge_down(gray, cx0, cx1, y_top - 0.3 * bh, y_bot + 0.6 * bh, 0.004 * fw)
        if b is None:
            b, basis = float(y_bot), "모델"
        bottoms.append(b)
    # 가로선은 한 줄: 위·아래 모두 높은 쪽(y가 작은 쪽) 눈썹 기준
    return min(tops), min(bottoms), basis


def _nostril_top(gray, regs, p) -> tuple[float, str]:
    """콧구멍의 가장 높은 위치: 코 아래 영역에서 가장 어두운 두 덩어리의 위 끝."""
    if "코아래" in regs:
        hull = regs["코아래"]
        x0, y0 = np.floor(hull.min(axis=0)).astype(int)
        x1, y1 = np.ceil(hull.max(axis=0)).astype(int)
        x0, y0 = max(0, x0), max(0, y0)
        crop = gray[y0:y1, x0:x1]
        if crop.size > 20:
            m = R.poly_mask(crop.shape, hull, hull=True, offset=(x0, y0)) > 0
            vals = crop[m]
            thr = np.percentile(vals, 8)
            dark = ((crop <= thr) & m).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
            if n > 1:
                order = np.argsort(st[1:, cv2.CC_STAT_AREA])[::-1][:2] + 1
                return float(y0 + min(st[i, cv2.CC_STAT_TOP] for i in order)), "영상"
    return float(p["코밑"][1]), "추정"


def _skin_union(frame) -> np.ndarray | None:
    m = frame.skin_mask
    b = frame.masks.get("몸피부")
    if m is None and b is None:
        return None
    if m is None:
        return b
    # 얼굴·몸 피부 경계(턱 밑)에서는 두 값이 모두 0.5 아래로 내려가 틈이 생기므로 더한다
    return m if b is None else np.clip(m + b, 0, 1)


def _row_edges(mask: np.ndarray, y: int, start_l: float, start_r: float,
               cx: float | None = None) -> tuple[float | None, float | None]:
    """y행에서 왼쪽 시작점부터 왼쪽으로, 오른쪽 시작점부터 오른쪽으로 마스크가 끝나는 x.

    시작점이 마스크 밖이면(턱이 좁아짐) cx 쪽으로 들어가며 마스크를 먼저 찾는다.
    """
    h, w = mask.shape[:2]
    if not 0 <= y < h:
        return None, None
    row = mask[y] > 0.5
    out = []
    for x, d in ((int(start_l), -1), (int(start_r), 1)):
        x = min(max(x, 0), w - 1)
        if not row[x] and cx is not None:
            while x != int(cx) and not row[x]:
                x -= d
        if not row[x]:
            out.append(None)
            continue
        while 0 <= x + d < w and row[x + d]:
            x += d
        out.append(float(x))
    return out[0], out[1]


def _jaw_neck_bend(mask, cx, y_from, y_to, start_l, start_r, fw) -> tuple[float, float, float] | None:
    """턱 라인이 목으로 꺾이는 점 (왼 x, 오른 x, 낮은 쪽 y). 한쪽씩 윤곽을 두 직선(턱·목)으로 나눠 맞춘다."""
    ys = np.arange(int(y_from), int(y_to), max(1, round(fw / 300)))   # 원본에서도 300행 안팎만 본다
    if ys.size < 10:
        return None
    L, Rr = [], []
    sl, sr = start_l, start_r
    for y in ys:
        l, r = _row_edges(mask, int(y), sl, sr, cx)
        L.append(np.nan if l is None else cx - l)
        Rr.append(np.nan if r is None else r - cx)
        # 다음 행은 이번 경계보다 조금 안쪽에서 시작 (턱이 좁아져도 따라가게)
        if l is not None:
            sl = min(l + 0.04 * fw, cx - 1)
        if r is not None:
            sr = max(r - 0.04 * fw, cx + 1)
    res = []
    for prof in (np.array(L), np.array(Rr)):
        ok = ~np.isnan(prof)
        yy, pp = ys[ok].astype(np.float64), prof[ok]
        if yy.size < 10:
            return None
        best, best_k = None, None
        for k in range(4, yy.size - 4):
            e = 0.0
            for seg_y, seg_p in ((yy[:k], pp[:k]), (yy[k:], pp[k:])):
                c = np.polyfit(seg_y, seg_p, 1)
                e += float(((np.polyval(c, seg_y) - seg_p) ** 2).sum())
            if best is None or e < best:
                best, best_k = e, k
        a1, b1 = np.polyfit(yy[:best_k], pp[:best_k], 1)
        a2, b2 = np.polyfit(yy[best_k:], pp[best_k:], 1)
        if not (a1 < -0.15 and abs(a2) < abs(a1) * 0.6):      # 턱은 빠르게 좁아지고 목은 거의 세로
            return None
        # 턱은 곡선이라 두 직선의 교점은 실제보다 위로 잡힌다. 목 아래쪽으로 목 직선을 다시 맞추고,
        # 아래에서 위로 올라가며 윤곽이 목 직선에서 벗어나기 직전 행을 꺾임점으로 한다.
        lo = best_k + (yy.size - best_k) * 3 // 10
        if yy.size - lo >= 4:
            a2, b2 = np.polyfit(yy[lo:], pp[lo:], 1)
        tol = max(2.0, 0.012 * fw)
        j = yy.size - 1
        while j - 1 >= 0 and abs(pp[j - 1] - (a2 * yy[j - 1] + b2)) <= tol:
            j -= 1
        res.append((float(a2 * yy[j] + b2), float(yy[j])))
    (dl, yl), (dr, yr) = res
    return cx - dl, cx + dr, max(yl, yr)


def _head_width(mask, cx, y0, y1) -> tuple[float, float] | None:
    """y0~y1 사이에서 마스크가 좌우로 가장 넓게 퍼진 x (머리카락 바깥 폭·두상 끝)."""
    h, w = mask.shape[:2]
    ya, yb = int(max(0, y0)), int(min(h, y1))
    if yb <= ya:
        return None
    band = mask[ya:yb] > 0.5
    cols = np.flatnonzero(band.any(axis=0))
    if cols.size == 0:
        return None
    # 중심선과 이어진 덩어리만: 중심에서 바깥으로 끊기지 않는 마지막 열
    filled = band.any(axis=0)
    l = int(cx)
    while l - 1 >= 0 and filled[l - 1]:
        l -= 1
    r = int(cx)
    while r + 1 < w and filled[r + 1]:
        r += 1
    return float(l), float(r)


def _shoulder_y(person, x, y_from) -> float | None:
    """x 열에서 y_from 아래로 내려가며 사람 영역(어깨)이 처음 시작되는 y."""
    h, w = person.shape[:2]
    xi = int(round(x))
    if not 0 <= xi < w:
        return None
    col = person[int(max(0, y_from)):, xi] > 0.5
    idx = np.flatnonzero(col)
    return None if idx.size == 0 else float(y_from + idx[0])


def _pair(cx: float, l: float | None, r: float | None) -> tuple[float, tuple[float, float]] | None:
    if l is None or r is None:
        return None
    return min(cx - l, r - cx), (l, r)


# ------------------------------------------------------------ 템플릿

def measure_template(frame, gender: str) -> Template:
    p, regs = frame.points, frame.regions
    gray = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fw = R.face_w(p)
    cx = center_x(p)
    t = Template(gender, cx)
    female = gender == "여"

    # ---- 가로선
    brows = _brow_lines(gray, regs, fw)
    eyes = {}
    for e in ("왼눈", "오른눈"):
        up = max(p.get(f"{e}_홍채위", p[f"{e}_윗꺼풀"])[1], p[f"{e}_윗꺼풀"][1])
        lo = p.get(f"{e}_밑라인", p.get(f"{e}_홍채아래", p[f"{e}_아랫꺼풀"]))[1]
        eyes[e] = (up, lo)
    big = max(eyes, key=lambda e: eyes[e][1] - eyes[e][0])     # 눈 큰 쪽 우선 (§5-2)
    corner_y = (p["왼눈_바깥꼭짓점"][1] + p["오른눈_바깥꼭짓점"][1]) / 2
    nostril, nb = _nostril_top(gray, regs, p)
    wing = max(p["왼콧볼"][1], p["오른콧볼"][1])
    if "코아래" in regs:
        wing = float(max(wing, np.percentile(regs["코아래"][:, 1], 85)))
    mouth = (p["윗입술_안쪽"][1] + p["아랫입술_안쪽"][1]) / 2 if "윗입술_안쪽" in p else \
        (p["왼입꼬리"][1] + p["오른입꼬리"][1]) / 2

    union = _skin_union(frame)
    chin, chin_b = float(p["턱끝"][1]), "모델"
    if frame.skin_mask is not None:
        col = frame.skin_mask[:, int(max(0, cx - 0.03 * fw)):int(cx + 0.03 * fw) + 1].mean(axis=1)
        ys = np.flatnonzero(col > 0.5)
        ys = ys[(ys > mouth) & (ys < chin + 0.15 * fw)]
        if ys.size:
            chin, chin_b = float(ys.max()), "영상"

    bend = None
    if union is not None:
        sl = p["왼볼_입높이"][0] + 0.15 * (cx - p["왼볼_입높이"][0])
        sr = p["오른볼_입높이"][0] - 0.15 * (p["오른볼_입높이"][0] - cx)
        bend = _jaw_neck_bend(union, cx, mouth, chin + 0.35 * fw, sl, sr, fw)
    if bend is None:
        t.notes.append("턱·목 꺾임점을 찾지 못해 턱선 점으로 추정")
        jl, jr = p["왼턱선"], p["오른턱선"]
        bend = (jl[0], jr[0], max(jl[1], jr[1]))
        bend_b = "추정"
    else:
        bend_b = "영상"

    # 얼굴 윤곽 at 입 높이
    contour = None
    if union is not None:
        sl = p["왼볼_입높이"][0] + 0.15 * (cx - p["왼볼_입높이"][0])
        sr = p["오른볼_입높이"][0] - 0.15 * (p["오른볼_입높이"][0] - cx)
        contour = _pair(cx, *_row_edges(union, int(round(mouth)), sl, sr, cx))
    contour_b = "영상"
    if contour is None:
        contour, contour_b = _pair(cx, p["왼볼_입높이"][0], p["오른볼_입높이"][0]), "모델"

    # 머리 폭 (여: 머리카락 바깥 / 남: 두상 끝 — 귀 끝은 아직 못 찾음)
    head = None
    head_mask = None
    for m in (frame.hair_mask, union):
        if m is not None:
            head_mask = m if head_mask is None else np.maximum(head_mask, m)
    if head_mask is not None:                                  # 옷이 섞이지 않게 머리카락·피부만
        if female:
            hl = _head_width(head_mask, cx, corner_y, chin)
        else:
            hl = _head_width(head_mask, cx, brows[0] if brows else corner_y, nostril)
        head = _pair(cx, *(hl or (None, None)))
    shoulder = None
    if head is not None and frame.person_mask is not None:
        ys = [_shoulder_y(frame.person_mask, x, chin + 0.05 * fw) for x in (cx - head[0], cx + head[0])]
        ys = [y for y in ys if y is not None]
        shoulder = max(ys) if ys else None                    # 낮은 쪽

    iris_in = None
    if "왼눈_홍채안쪽" in p and "오른눈_홍채안쪽" in p:
        iris_in = _pair(cx, p["왼눈_홍채안쪽"][0], p["오른눈_홍채안쪽"][0])

    H, V = t.h, t.v
    if female:
        if brows:
            H += [Line("A", "눈썹 위", brows[0]), Line("B", "눈썹 아래 끝", brows[1], brows[2])]
        H += [Line("C", "윗눈꺼풀", min(e[0] for e in eyes.values())),
              Line("D", "눈 바깥 꼭짓점", corner_y, "영상"),
              Line("E", "아랫눈꺼풀", max(e[1] for e in eyes.values()), "영상"),
              Line("F", "콧구멍 위", nostril, nb), Line("G", "콧볼 하단", wing),
              Line("H", "입술 만나는 선", mouth), Line("I", "턱·목 꺾임", bend[2], bend_b),
              Line("J", "턱 끝", chin, chin_b)]
        if shoulder is not None:
            if abs(shoulder - chin) < 0.02 * fw:
                t.notes.append("K선이 J선과 겹쳐 한 줄로 씀")
            else:
                H.append(Line("K", "머리 폭이 어깨에 닿는 높이", shoulder, "영상"))
        V.append(Line("중심", "중심선", cx))
        V += [Line("눈안쪽", "눈 안쪽 끝", x) for x in (p["왼눈_안쪽꼭짓점"][0], p["오른눈_안쪽꼭짓점"][0])]
        if iris_in:
            V.append(Line("주황", "홍채 안쪽", iris_in[0], "모델", True, iris_in[1]))
        if contour:
            V.append(Line("초록", "입 높이 얼굴 윤곽", contour[0], contour_b, True, contour[1]))
        jp = _pair(cx, bend[0], bend[1])
        if jp:
            V.append(Line("자주", "턱·목 경계", jp[0], bend_b, True, jp[1]))
        if head:
            V.append(Line("보라", "머리카락 바깥 폭", head[0], "영상", True, head[1]))
    else:
        if brows:
            H += [Line("1", "눈썹 위", brows[0]), Line("2", "눈썹 몸통 아래 끝", brows[1], brows[2])]
        H += [Line("3", "눈동자 위 (큰 눈)", eyes[big][0]),
              Line("4", "눈 바깥 꼭짓점", corner_y, "영상"),
              Line("5", "눈동자 아래 (큰 눈)", eyes[big][1], "영상"),
              Line("6", "콧구멍 위", nostril, nb), Line("7", "콧볼 하단", wing),
              Line("9", "입술 만나는 선", mouth), Line("10", "턱·목 꺾임", bend[2], bend_b),
              Line("11", "턱 끝", chin, chin_b)]
        if shoulder is not None:
            H.append(Line("12", "머리 폭이 어깨에 닿는 높이", shoulder, "영상"))
        t.notes.append("귀 끝선(8번)은 아직 찾지 않음")
        V.append(Line("A", "중심선", cx))
        V += [Line("B", "눈 안쪽 끝", x) for x in (p["왼눈_안쪽꼭짓점"][0], p["오른눈_안쪽꼭짓점"][0])]
        if iris_in:
            V.append(Line("C", "홍채 안쪽", iris_in[0], "모델", True, iris_in[1]))
        jp = _pair(cx, bend[0], bend[1])
        if jp:
            V.append(Line("D", "턱·목 경계", jp[0], bend_b, True, jp[1]))
        if contour:
            V.append(Line("E", "입 높이 얼굴 윤곽", contour[0], contour_b, True, contour[1]))
        if head:
            V.append(Line("F", "두상 끝 (머리 폭)", head[0], "영상", True, head[1]))
    return t
