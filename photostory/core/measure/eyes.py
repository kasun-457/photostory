"""정밀 측정 P3: 눈 (가이드 §4).

- 눈 바깥 꼭짓점: 위·아래 눈꺼풀이 만나는 끝. 모델 점은 실제 꼭짓점보다 아래·안쪽에 찍히므로
  원본 해상도에서 '어두운 눈 틈'이 바깥쪽으로 어디까지 이어지는지 따라가서 다시 찾는다.
- 아랫눈꺼풀 밑라인(대체 기준): 홍채 아래 끝과 아랫눈꺼풀이 만나는 경계.
  반사판 때문에 그 아래가 밝게 뜨는 경우가 있어, 홍채에서 내려가며 만나는 '첫' 경계를 쓴다.
- 둘 다 확신도(0~1)를 함께 낸다. 수평 단계는 꼭짓점 확신도가 낮으면 밑라인을 쓴다.
기준값은 공개 사진으로 잡은 값이라 실제 사진관 원본으로 다시 맞춰야 한다 (확실하지 않음).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

CORNER_OK = 0.5      # 이 확신도 이상이면 꼭짓점으로 수평을 잡는다
LID_OK = 0.4
LID_AGREE_DEG = 1.0  # 밑라인 각도와 눈 끝(모델 점) 각도가 이보다 다르면 밑라인을 쓰지 않음


@dataclass
class Found:
    point: tuple[float, float]
    confidence: float


def _rectified(gray: np.ndarray, origin, u, v, ts: np.ndarray, ss: np.ndarray) -> np.ndarray:
    """origin + u·t + v·s 위치의 밝기를 [s, t] 격자로 뽑는다."""
    T, S = np.meshgrid(ts, ss)
    mx = (origin[0] + u[0] * T + v[0] * S).astype(np.float32)
    my = (origin[1] + u[1] * T + v[1] * S).astype(np.float32)
    return cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _runs(col: np.ndarray) -> list[tuple[int, int]]:
    """True 구간들 [(시작, 끝)]."""
    out, start = [], None
    for k, x in enumerate(col):
        if x and start is None:
            start = k
        elif not x and start is not None:
            out.append((start, k - 1))
            start = None
    if start is not None:
        out.append((start, len(col) - 1))
    return out


def refine_outer_corner(gray: np.ndarray, corner, inner) -> Found:
    """gray: float32 0~255 원본 해상도. corner/inner: 모델이 찍은 바깥·안쪽 꼭짓점.

    눈 축을 따라 바깥쪽으로 가며 세로 단면을 본다. 꼭짓점 안쪽에서는 윗눈꺼풀 선과 아랫눈꺼풀 선이
    흰자를 사이에 두고 두 줄로 보이고, 꼭짓점에서 한 줄로 합쳐진다. 처음 합쳐지는 곳이 꼭짓점이다.
    그 바깥으로 이어지는 아이라인·속눈썹 꼬리는 한 줄이라서 따라가지 않는다 (가이드 §4).
    """
    c, i = np.asarray(corner, np.float64), np.asarray(inner, np.float64)
    eye_w = float(np.hypot(*(c - i)))
    if eye_w < 8:
        return Found(tuple(corner), 0.0)
    u = (c - i) / eye_w                       # 바깥쪽
    v = np.array([-u[1], u[0]])               # 눈 축에 수직, 위쪽(+)으로 맞춘다
    if v[1] > 0:
        v = -v
    step = max(0.5, eye_w / 300)              # 원본 4472px에서 눈 폭 약 300px → 1px 간격
    ts = np.arange(-0.20 * eye_w, 0.40 * eye_w, step)
    tall = np.arange(-0.22 * eye_w, 0.22 * eye_w, step)
    P = _rectified(gray, c, u, v, ts, tall)
    P = cv2.GaussianBlur(P, (0, 0), max(0.6, 0.006 * eye_w / step))
    skin = float(np.median(P))
    # 같은 위치의 위·아래(세로로 긴 창의 중간값)보다 얼마나 어두운가 → 그늘진 눈가는 선으로 안 봄
    ref = np.median(P, axis=0)
    win = (tall >= -0.08 * eye_w) & (tall <= 0.14 * eye_w)   # 모델 점은 실제보다 아래라 위를 더 봄
    D = ref[None, :] - P[win]
    sw = tall[win]
    line = float(np.percentile(D[:, ts <= 0.1 * eye_w], 98)) if D.size else 0.0
    if line <= 3.0 or skin <= 1.0:
        return Found(tuple(corner), 0.0)
    dark = D >= 0.45 * line
    min_len = max(1, round(0.004 * eye_w / step))
    runs = [[r for r in _runs(dark[:, j]) if r[1] - r[0] + 1 >= min_len] for j in range(len(ts))]
    hold = max(3, round(0.02 * eye_w / step))   # 합쳐진 상태가 이만큼 이어져야 꼭짓점으로 인정
    two_cols, j_c = 0, None
    for j in range(len(ts)):
        if len(runs[j]) >= 2:
            two_cols += 1
            continue
        if two_cols >= 3 and all(len(runs[k]) <= 1 for k in range(j, min(len(ts), j + hold))):
            j_c = j
            break
    if j_c is None:
        return Found(tuple(corner), 0.0)
    # 선 굵기·흐림 때문에 두 줄은 실제 꼭짓점보다 조금 안쪽에서 붙어 보인다.
    # 합쳐지기 직전 구간에서 윗선·아랫선 가운데를 직선으로 맞추고, 두 직선이 만나는 곳을 꼭짓점으로 한다.
    K = max(4, round(0.06 * eye_w / step))
    cols = [j for j in range(max(0, j_c - K), j_c) if len(runs[j]) >= 2]
    t_c = float(ts[j_c])
    if runs[j_c]:
        r0, r1 = runs[j_c][0]
        s_c = float((sw[r0] + sw[r1]) / 2)
    else:                                     # 선이 바로 끝남: 직전 두 줄의 가운데
        (a0, a1), (b0, b1) = runs[j_c - 1][:2]
        s_c = float((sw[a1] + sw[b0]) / 2)
    if len(cols) >= 3:
        tt = ts[cols]
        lo_c = np.array([(sw[runs[j][0][0]] + sw[runs[j][0][1]]) / 2 for j in cols])
        up_c = np.array([(sw[runs[j][-1][0]] + sw[runs[j][-1][1]]) / 2 for j in cols])
        (au, bu), (al, bl) = np.polyfit(tt, up_c, 1), np.polyfit(tt, lo_c, 1)
        if au - al < -1e-6:                   # 윗선은 내려오고 아랫선은 올라와야 만난다
            t_x = (bl - bu) / (au - al)
            if t_c <= t_x <= t_c + 0.10 * eye_w:
                t_c, s_c = float(t_x), float(au * t_x + bu)
    p = c + u * t_c + v * s_c
    contrast = min(1.0, (line / skin) / 0.15)
    near = 1.0 if -0.10 * eye_w <= t_c <= 0.25 * eye_w else 0.3
    return Found((float(p[0]), float(p[1])), round(contrast * near, 3))


def lower_lid(gray: np.ndarray, iris_center, iris_bottom, iris_r: float) -> Found:
    """홍채 아래 끝과 아랫눈꺼풀 경계. 홍채 가운데 띠의 세로 밝기에서 '어둠 → 밝음' 첫 경계."""
    cx, cy = iris_center
    r = max(2.0, float(iris_r))
    h, w = gray.shape[:2]
    x0, x1 = int(max(0, cx - 0.3 * r)), int(min(w, cx + 0.3 * r + 1))
    y0, y1 = int(max(0, cy + 0.4 * r)), int(min(h, cy + 1.6 * r))
    if x1 - x0 < 1 or y1 - y0 < 4:
        return Found(tuple(iris_bottom), 0.0)
    prof = gray[y0:y1, x0:x1].mean(axis=1)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1).astype(np.float32), (1, 0), max(0.6, 0.06 * r)).ravel()
    g = np.gradient(prof)
    gmax = float(g.max())
    if gmax <= 0.5:
        return Found(tuple(iris_bottom), 0.0)
    # 첫 번째로 충분히 센 오르막의 꼭대기
    k = int(np.argmax(g >= 0.5 * gmax))
    while k + 1 < len(g) and g[k + 1] >= g[k]:
        k += 1
    y = y0 + k
    conf = min(1.0, gmax / 4.0) * (1.0 if abs(y - iris_bottom[1]) < 0.5 * r else 0.5)
    return Found((float(cx), float(y)), round(conf, 3))


def angle(a, b) -> float:
    """사진 왼쪽 점 a → 오른쪽 점 b 선의 기울기 (도). +면 오른쪽이 아래."""
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def measure_eyes(rgb: np.ndarray, points: dict) -> dict:
    """prepare용. 다시 찾은 점과 수평 기준을 돌려준다.

    반환: {"points": {...새 표준 점...}, "level": {"basis", "deg", "corner_conf", "lid_conf"}}
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    new: dict[str, tuple[float, float]] = {}
    cc, lc = {}, {}
    for eye in ("왼눈", "오른눈"):
        f = refine_outer_corner(gray, points[f"{eye}_바깥꼭짓점"], points[f"{eye}_안쪽꼭짓점"])
        new[f"{eye}_바깥꼭짓점_모델"] = points[f"{eye}_바깥꼭짓점"]
        if f.confidence > 0:
            new[f"{eye}_바깥꼭짓점"] = f.point
        cc[eye] = f.confidence
        c, b = points.get(f"{eye}_홍채중심"), points.get(f"{eye}_홍채아래")
        if c is not None and b is not None:
            rr = [np.hypot(points[k][0] - c[0], points[k][1] - c[1])
                  for k in (f"{eye}_홍채위", f"{eye}_홍채아래", f"{eye}_홍채안쪽", f"{eye}_홍채바깥") if k in points]
            g = lower_lid(gray, c, b, float(np.mean(rr)) if rr else 0.0)
            if g.confidence > 0:
                new[f"{eye}_밑라인"] = g.point
            lc[eye] = g.confidence
        else:
            lc[eye] = 0.0
    corner_conf, lid_conf = min(cc.values()), min(lc.values())
    if corner_conf >= CORNER_OK:
        basis = "꼭짓점"
    elif (lid_conf >= LID_OK and "왼눈_밑라인" in new and "오른눈_밑라인" in new
          and abs(angle(new["왼눈_밑라인"], new["오른눈_밑라인"])
                  - angle(new["왼눈_바깥꼭짓점_모델"], new["오른눈_바깥꼭짓점_모델"])) <= LID_AGREE_DEG):
        # 가이드: 밑라인으로 맞춘 뒤 눈 끝도 함께 맞는지 본다 → 크게 어긋나면 밑라인을 믿지 않는다
        basis = "밑라인"
    else:
        basis = "꼭짓점(모델 점)"
        for eye in ("왼눈", "오른눈"):           # 둘 다 애매하면 모델 점 그대로 (보수적)
            new[f"{eye}_바깥꼭짓점"] = new[f"{eye}_바깥꼭짓점_모델"]
    return {"points": new, "level": {"basis": basis, "corner_conf": corner_conf, "lid_conf": lid_conf}}
