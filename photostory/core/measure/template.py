"""측정 안내선 (가이드 §5 여자 5-1 / 남자 5-2 템플릿).

수평 맞춘 뒤의 이미지에서 잰다. 결과는 가로선(y)·세로선(x) 목록이다.
- 세로 대칭쌍(얼굴 윤곽·턱목 경계·머리 폭 등): 중심선에서 좌우를 재고 '좁은 쪽' 거리를 양쪽에 쓴다 (§5-0).
- 각 선은 어떻게 쟀는지(basis)를 함께 낸다: "모델"(얼굴 점 그대로), "영상"(사진 경계에서 다시 찾음),
  "추정"(찾지 못해 근사). 2단계 대칭 보정은 '실제 경계 − 안내선'을 보정량으로 쓴다.

- 귀 끝선(남자 8번)·귀 끝 머리 폭(F): 얼굴 윤곽 바깥의 피부 덩어리(귀)에서 찾는다. 못 찾으면 생략·두상 끝.
- 쌍꺼풀선(§5-3): 두 눈 모두 주름을 찾으면 더 또렷한(또는 지정한) 눈의 주름 가장 높은 곳.
- 눈 윤곽 위 선(§5-5): 윗눈꺼풀이 홍채를 덮어 '홍채 위 경계'와 '눈 윤곽 위'의 차이가 클 때만.
- 그림자 턱선(§5-4): 바깥 윤곽으로 턱·목 꺾임을 못 찾으면, 얼굴 안쪽 그림자가 시작되는 위쪽 경계가
  목 옆선과 만나는 점. 가이드도 자동 추적은 참고만 한다고 해서 basis '그림자'로 두고,
  대칭·목 보정에는 쓰지 않는다 (안내선으로만 보여 줌).
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


def _ears(skin: np.ndarray, oval: np.ndarray, y0: float, y1: float, fw: float):
    """얼굴 윤곽 바깥에 붙은 피부 덩어리 = 귀. 쪽마다 (아래 끝 y, 바깥 x) 또는 None.

    얼굴 윤곽(모델 다각형)은 실제 얼굴보다 조금 안쪽이라 얼굴 폭 4%만큼 넓혀서 뺀다.
    """
    h, w = skin.shape[:2]
    ya, yb = int(max(0, y0)), int(min(h, y1))
    if yb - ya < 5:
        return None, None
    face = R.dilate(R.poly_mask((h, w), oval), 0.04 * fw) > 0
    band = (skin[ya:yb] > 0.5) & ~face[ya:yb]
    band = cv2.morphologyEx(band.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
    ox0, ox1 = float(oval[:, 0].min()), float(oval[:, 0].max())
    out = []
    for side in ("왼", "오른"):
        best = None
        for i in range(1, n):
            x, y, bw, bh, area = st[i]
            if area < 0.002 * fw * fw or bh < 0.06 * fw or bw > 0.25 * fw:
                continue
            inner = x + bw if side == "왼" else x                     # 얼굴 쪽 끝
            if side == "왼" and not (x + bw <= ox0 + 0.08 * fw and inner >= ox0 - 0.10 * fw):
                continue
            if side == "오른" and not (x >= ox1 - 0.08 * fw and inner <= ox1 + 0.10 * fw):
                continue
            if best is None or area > st[best][4]:
                best = i
        if best is None:
            out.append(None)
            continue
        x, y, bw, bh, _ = st[best]
        out.append((float(ya + y + bh - 1), float(x if side == "왼" else x + bw - 1)))
    return out[0], out[1]


def _shadow_jaw(gray: np.ndarray, union: np.ndarray, cx: float, mouth: float, chin: float, fw: float):
    """얼굴살로 바깥 윤곽이 볼에서 목까지 한 줄일 때 (§5-4): 그림자 턱선과 목 옆선이 만나는 점.

    목 옆선 = 턱 끝 아래 바깥 윤곽의 x (거의 세로). 그 x 둘레 몇 열에서 입 높이 아래로 내려가며
    '밝음 → 어둠'이 가장 센 곳 = 그림자가 시작되는 위쪽 경계 (어두운 중심이 아님, 가이드 경고).
    (왼 x, 오른 x, 낮은 쪽 y) 또는 None.
    """
    h, w = gray.shape[:2]
    hp = gray - cv2.GaussianBlur(gray, (0, 0), max(2.0, 0.05 * fw))        # 큰 반경 블러를 뺀 하이패스
    hp = cv2.GaussianBlur(hp, (0, 0), max(0.8, 0.006 * fw))
    res = []
    for d in (-1, 1):
        xs = []
        for y in np.linspace(chin + 0.05 * fw, chin + 0.25 * fw, 8):
            l, r = _row_edges(union, int(round(y)), cx - 0.2 * fw, cx + 0.2 * fw, cx)
            e = l if d < 0 else r
            if e is not None:
                xs.append(e)
        if len(xs) < 4 or np.std(xs) > 0.03 * fw:
            return None
        nx = float(np.median(xs)) - d * 0.03 * fw                          # 윤곽 바로 안쪽
        ya, yb = int(max(0, mouth + 0.05 * fw)), int(min(h, chin + 0.10 * fw))
        if yb - ya < 5:
            return None
        hits, strengths = [], []
        for x in np.linspace(nx - d * 0.04 * fw, nx, 5):
            xi = int(round(x))
            if not 0 <= xi < w:
                continue
            g = np.gradient(hp[ya:yb, xi])
            k = int(np.argmin(g))                                           # 밝음 → 어둠
            hits.append(ya + k)
            strengths.append(-g[k])
        if len(hits) < 3 or np.median(strengths) < 0.6 or np.std(hits) > 0.04 * fw:
            return None
        res.append((nx, float(np.median(hits))))
    (xl, yl), (xr, yr) = res
    return xl, xr, max(yl, yr)


def _pair(cx: float, l: float | None, r: float | None) -> tuple[float, tuple[float, float]] | None:
    if l is None or r is None:
        return None
    return min(cx - l, r - cx), (l, r)


# ------------------------------------------------------------ 템플릿

def eye_opening(p, e: str) -> tuple[float, float]:
    """(눈동자 위 = 윗눈꺼풀이 홍채를 가리는 곳, 눈동자 아래 = 아랫눈꺼풀 밑라인) y."""
    up = max(p.get(f"{e}_홍채위", p[f"{e}_윗꺼풀"])[1], p[f"{e}_윗꺼풀"][1])
    lo = p.get(f"{e}_밑라인", p.get(f"{e}_홍채아래", p[f"{e}_아랫꺼풀"]))[1]
    return up, lo



def measure_template(frame, gender: str) -> Template:
    p, regs = frame.points, frame.regions
    gray = cv2.cvtColor(frame.image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fw = R.face_w(p)
    cx = center_x(p)
    t = Template(gender, cx)
    female = gender == "여"

    # ---- 가로선
    brows = _brow_lines(gray, regs, fw)
    eyes = {e: eye_opening(p, e) for e in ("왼눈", "오른눈")}
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
    bend_b = "영상"
    if bend is None and union is not None:
        bend = _shadow_jaw(gray, union, cx, mouth, chin, fw)
        if bend is not None:
            bend_b = "그림자"
            t.notes.append("턱·목 꺾임: 바깥 윤곽에 꺾임이 없어 얼굴 안쪽 그림자 턱선으로 그음 (추측, 보정에는 안 씀)")
    if bend is None:
        t.notes.append("턱·목 꺾임점을 찾지 못해 턱선 점으로 추정")
        jl, jr = p["왼턱선"], p["오른턱선"]
        bend = (jl[0], jr[0], max(jl[1], jr[1]))
        bend_b = "추정"

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
    # 귀 (남자 8번·F): 귀가 두상보다 튀어나왔으면 귀 끝, 좌우 중 덜 튀어나온 쪽 (§5-2)
    ear_l = ear_r = None
    if not female and frame.skin_mask is not None and "얼굴윤곽" in regs:
        ear_l, ear_r = _ears(np.maximum(frame.skin_mask, union) if union is not None else frame.skin_mask,
                             regs["얼굴윤곽"], (brows[0] if brows else corner_y), mouth, fw)
    ear_head = None
    if ear_l and ear_r:
        ep = _pair(cx, ear_l[1], ear_r[1])
        if head is None or ep[0] > head[0]:
            ear_head = ep
    shoulder = None
    if ear_head is not None:
        head = ear_head
    if head is not None and frame.person_mask is not None:
        ys = [_shoulder_y(frame.person_mask, x, chin + 0.05 * fw) for x in (cx - head[0], cx + head[0])]
        ys = [y for y in ys if y is not None]
        shoulder = max(ys) if ys else None                    # 낮은 쪽

    # 눈 윤곽 위 (§5-5): 큰 눈 기준. 홍채 위 경계와 눈 윤곽(윗속눈썹 라인) 가장 높은 곳의 차이가 클 때만
    lid_top = None
    if big in regs:
        top = float(regs[big][:, 1].min())
        if eyes[big][0] - top > 0.012 * fw:
            lid_top = top
    # 쌍꺼풀 (§5-3): 두 눈 모두 주름이 있을 때만
    crease_y = None
    from ...steps.crease import measure_crease          # 늦게 읽기 (단계 모듈이 이 모듈을 읽음)
    cr = {e: measure_crease(gray, regs, e, b) for e, b in (("왼눈", "왼눈썹"), ("오른눈", "오른눈썹"))}
    if cr["왼눈"] is not None and cr["오른눈"] is not None:
        ref = getattr(frame, "crease_ref", None) or "auto"
        if ref not in cr:
            ref = max(cr, key=lambda e: cr[e].clarity)
        crease_y = cr[ref].y

    iris_in = None
    if "왼눈_홍채안쪽" in p and "오른눈_홍채안쪽" in p:
        iris_in = _pair(cx, p["왼눈_홍채안쪽"][0], p["오른눈_홍채안쪽"][0])

    H, V = t.h, t.v
    if female:
        if brows:
            H += [Line("A", "눈썹 위", brows[0]), Line("B", "눈썹 아래 끝", brows[1], brows[2])]
        if crease_y is not None:
            H.append(Line("쌍꺼풀", "쌍꺼풀 주름 가장 높은 곳", crease_y, "영상"))
        if lid_top is not None:
            H.append(Line("눈윤곽위", "눈 윤곽 위 (안검하수)", lid_top))
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
        if crease_y is not None:
            H.append(Line("2-1", "쌍꺼풀 주름 가장 높은 곳", crease_y, "영상"))
        if lid_top is not None:
            H.append(Line("3-0", "눈 윤곽 위 (안검하수)", lid_top))
        H += [Line("3", "눈동자 위 (큰 눈)", eyes[big][0]),
              Line("4", "눈 바깥 꼭짓점", corner_y, "영상"),
              Line("5", "눈동자 아래 (큰 눈)", eyes[big][1], "영상"),
              Line("6", "콧구멍 위", nostril, nb), Line("7", "콧볼 하단", wing),
              Line("9", "입술 만나는 선", mouth), Line("10", "턱·목 꺾임", bend[2], bend_b),
              Line("11", "턱 끝", chin, chin_b)]
        if shoulder is not None:
            H.append(Line("12", "머리 폭이 어깨에 닿는 높이", shoulder, "영상"))
        ears = [e for e in (ear_l, ear_r) if e is not None]
        if not ears:
            t.notes.append("귀 끝을 찾지 못해 8번 선 생략, F는 두상 끝")
        else:
            ey = float(np.mean([e[0] for e in ears]))
            if abs(ey - wing) < 0.02 * fw:
                t.notes.append("귀 끝이 7번 선과 겹쳐 8번 선 생략")
            else:
                H.append(Line("8", "귀 끝", ey, "영상"))
                H.sort(key=lambda l: l.pos)
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
            V.append(Line("F", "귀 끝 (머리 폭)" if ear_head else "두상 끝 (머리 폭)", head[0], "영상", True, head[1]))
    return t
