"""2단계: 안경 (P5 짝 정렬 + 12번 얹기). 원본에 안경테를 그린 뒤 사진 전체를 돌리고·키우고·옮겨
'안경 쓴 사진'을 만든다."""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from .glasses_data import draw_frame, glasses_photo
from photostory.core.glasses import prepare_glasses
from photostory.core.prepare import analyze
from photostory.core.processor import Processor
from photostory.core.session import Customer, Session, Stroke
from photostory.core.specs import SpecRegistry

OFF = {"skin", "tone", "sharpen", "catchlight", "symmetry", "neck", "hair_tidy", "hair_bright", "shoulder"}


def _save(img, name):
    path = Path(tempfile.mkdtemp()) / name
    cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(str(path))
    return path


def _render(hub, orig, glass=None, glasses=True, strokes=()):
    s = Session(Customer(name="a", purposes=["passport"], glasses=glasses), original=_save(orig, "a_원본.png"),
                glasses_photo=None if glass is None else _save(glass, "a_안경.png"), glasses_strokes=list(strokes))
    s.disabled_steps = set(OFF)
    proc = Processor(hub, SpecRegistry())
    proc.start(s)
    f = proc.render(s, preview=False).frame
    proc.clear()
    return f


def test_frame_extracted_in_original_coords(hub, portrait):
    a = analyze("", hub, rgb=portrait)
    g_img, truth = glasses_photo(portrait, a.points)
    cut = prepare_glasses(a, analyze("", hub, rgb=g_img))
    assert cut is not None and cut["align_score"] > 0.9
    x0, y0 = cut["offset"]
    h, w = cut["alpha"].shape
    pred, tr = cut["alpha"] > 0.5, truth[y0:y0 + h, x0:x0 + w] > 0
    assert (pred & tr).sum() / tr.sum() > 0.9                # 그린 테를 거의 다 덮음
    assert (pred & tr).sum() / (pred | tr).sum() > 0.55      # 테 둘레를 너무 넓게 잡지 않음


def _dark_diff(f_on, f_off):
    a, b = f_on.outputs["retouched"].astype(int), f_off.outputs["retouched"].astype(int)
    return (b.sum(axis=2) - a.sum(axis=2)) > 150              # 얹은 테 (어두워진 곳)


@pytest.fixture(scope="module")
def placed(hub, portrait):
    a = analyze("", hub, rgb=portrait)
    g_img, _ = glasses_photo(portrait, a.points)
    on = _render(hub, portrait, g_img)
    off = _render(hub, portrait, glasses=False)
    return on, off


def test_frame_placed_centered_level(placed):
    on, off = placed
    assert "glasses" in on.outputs, on.notes
    d = _dark_diff(on, off)
    assert d.sum() > 500
    cx = on.outputs["guides"]["center_x"]
    cols = np.nonzero(d.any(axis=0))[0]
    assert abs((cols[0] + cols[-1]) / 2 - cx) <= 1.5          # 바깥 끝이 중심선 대칭
    ys_l = np.nonzero(d[:, :int(cx)].any(axis=1))[0]
    ys_r = np.nonzero(d[:, int(cx):].any(axis=1))[0]
    assert abs(ys_l.max() - ys_r.max()) <= 1.5                 # 렌즈 아래 수평


def test_eyes_come_from_original(placed):
    on, off = placed
    for e in ("왼눈", "오른눈"):
        x, y = (round(v) for v in on.points[f"{e}_홍채중심"])
        a = on.outputs["retouched"][y - 2:y + 3, x - 2:x + 3].astype(int)
        b = off.outputs["retouched"][y - 2:y + 3, x - 2:x + 3].astype(int)
        assert np.abs(a - b).max() <= 2, e


def test_crooked_frame_is_leveled(hub, portrait):
    """얼굴에 3° 비뚤게 쓴 안경도 렌즈 아래를 수평으로 앉힌다."""
    a = analyze("", hub, rgb=portrait)
    p = a.points
    _, m = draw_frame(portrait, p)
    h, w = portrait.shape[:2]
    c = ((p["왼눈_안쪽꼭짓점"][0] + p["오른눈_안쪽꼭짓점"][0]) / 2, p["왼눈_바깥꼭짓점"][1])
    m = cv2.warpAffine(m, cv2.getRotationMatrix2D(c, 3.0, 1.0), (w, h), flags=cv2.INTER_NEAREST)
    g = portrait.copy()
    g[m > 0] = (25, 25, 30)
    on = _render(hub, portrait, g)
    off = _render(hub, portrait, glasses=False)
    assert any("기울기" in x for x in on.outputs["glasses"]["moved"]), on.outputs["glasses"]
    d = _dark_diff(on, off)
    cx = on.outputs["guides"]["center_x"]
    ys_l = np.nonzero(d[:, :int(cx)].any(axis=1))[0]
    ys_r = np.nonzero(d[:, int(cx):].any(axis=1))[0]
    assert abs(ys_l.max() - ys_r.max()) <= 2


def test_no_glasses_photo_notes(hub, portrait):
    f = _render(hub, portrait, None, glasses=True)
    assert "glasses" not in f.outputs and any("안경" in n for n in f.notes)


def test_restore_brush(hub, portrait):
    """복원 붓: 오른쪽 버튼(원본으로)은 테를 지우고, 왼쪽 버튼(안경 사진에서)은 못 뗀 부분을 살린다."""
    a = analyze("", hub, rgb=portrait)
    p = a.points
    h, w = portrait.shape[:2]
    # 자동으로는 버려지는 작은 밝은 점(은색 힌지 흉내)을 원본 좌표에 찍은 뒤 안경 사진을 만든다
    hx, hy = p["왼눈_바깥꼭짓점"][0] - 0.02 * w, p["왼눈_바깥꼭짓점"][1] - 0.012 * h
    with_dot = portrait.copy()
    cv2.circle(with_dot, (round(hx), round(hy)), 3, (240, 240, 245), -1)
    g_img, _ = glasses_photo(with_dot, p)
    off = _render(hub, portrait, glasses=False)
    on = _render(hub, portrait, g_img)
    # 원본 좌표의 점 → 결과 좌표 (수평 회전 반영)
    keys = [k for k in ("코끝", "턱끝", "왼눈_바깥꼭짓점", "오른눈_바깥꼭짓점", "왼입꼬리", "오른입꼬리")]
    S, _ = cv2.estimateAffinePartial2D(np.float32([p[k] for k in keys]), np.float32([on.points[k] for k in keys]))
    qx, qy = S @ np.array([hx, hy, 1.0])
    d_on = _dark_diff(on, off)
    cx = on.outputs["guides"]["center_x"]
    ey = on.points["왼눈_바깥꼭짓점"][1] / h
    erase = Stroke("original", 0.06, [(0.05, ey), (cx / w - 0.02, ey)])     # 사진 왼쪽 테 전체를 원본으로
    restore = Stroke("glasses", 6 / w, [(qx / w, qy / h)])                 # 그 뒤 힌지 점만 안경 사진에서
    br = _render(hub, portrait, g_img, strokes=[erase, restore])
    d_br = _dark_diff(br, off)
    left = slice(int(0.1 * w), int(cx - 0.03 * w))
    assert d_on[:, left].sum() > 100 and d_br[:, left].sum() < 0.1 * d_on[:, left].sum()
    right = slice(int(cx + 0.03 * w), w)
    assert d_br[:, right].sum() > 0.8 * d_on[:, right].sum()                # 오른쪽 테는 그대로
    x, y = round(qx), round(qy)
    got = br.outputs["retouched"][y - 1:y + 2, x - 1:x + 2].astype(int).mean()
    base = off.outputs["retouched"][y - 1:y + 2, x - 1:x + 2].astype(int).mean()
    assert got > base + 30, (got, base)                        # 밝은 힌지 점이 살아남
    auto = on.outputs["retouched"][y - 1:y + 2, x - 1:x + 2].astype(int).mean()
    assert auto < got - 30, (auto, got)                        # 자동으로는 안 붙었던 점
