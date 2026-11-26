"""2단계 2-7: 쌍꺼풀 좌우 맞춤(§5-3), 콧볼 옆 그림자(§8-7)."""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from photostory.core.measure import regions as R
from photostory.core.prepare import analyze
from photostory.core.processor import Processor
from photostory.core.session import Customer, Session
from photostory.core.specs import SpecRegistry

ALL = {"level", "skin", "catchlight", "sharpen", "tone", "nose_shadow", "symmetry", "crease", "neck",
       "hair_tidy", "glasses", "hair_bright", "shoulder"}


def _render(hub, img, keep, crease_ref="auto"):
    path = Path(tempfile.mkdtemp()) / "a.png"
    cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(str(path))
    s = Session(Customer(name="a", purposes=["passport"]), original=path, crease_ref=crease_ref)
    s.disabled_steps = ALL - set(keep)
    proc = Processor(hub, SpecRegistry())
    proc.start(s)
    f = proc.render(s, preview=False).frame
    proc.clear()
    return f


# ------------------------------------------------------------ 쌍꺼풀

def _lid_brow(a, eye, brow, x):
    em = R.poly_mask(a.image.shape, a.regions[eye]) > 0
    bm = R.poly_mask(a.image.shape, a.regions[brow], hull=True) > 0
    return int(np.nonzero(em[:, x])[0].min()), int(np.nonzero(bm[:, x])[0].max())


@pytest.fixture(scope="module")
def creased(hub, portrait):
    """2배로 키운 얼굴에 쌍꺼풀 주름을 그린다: 왼눈은 눈꺼풀과 눈썹 사이 35%, 오른눈은 60% 높이."""
    big = cv2.resize(portrait, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    a = analyze("", hub, rgb=big)
    img = big.copy()
    for eye, brow, frac in (("왼눈", "왼눈썹", 0.35), ("오른눈", "오른눈썹", 0.60)):
        e = a.regions[eye]
        x0, x1 = int(e[:, 0].min() + 0.1 * np.ptp(e[:, 0])), int(e[:, 0].max() - 0.1 * np.ptp(e[:, 0]))
        pts = []
        for x in range(x0, x1 + 1):
            lid, bb = _lid_brow(a, eye, brow, x)
            pts.append((x, lid - frac * (lid - bb)))
        line = np.zeros(img.shape[:2], np.uint8)
        cv2.polylines(line, [np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)], False, 255, 2, cv2.LINE_AA)
        a_ = cv2.GaussianBlur(line.astype(np.float32) / 255, (0, 0), 0.7)[..., None]
        img = np.clip(img * (1 - 0.45 * a_), 0, 255).astype(np.uint8)
    return img


def test_crease_heights_matched(hub, creased):
    f = _render(hub, creased, {"crease"})
    c = f.outputs["crease"]
    assert c["왼눈"] and c["오른눈"], c
    assert abs(c["왼눈"]["height"] - c["오른눈"]["height"]) >= 5, c
    assert c["after_diff"] is not None and c["after_diff"] <= 2.5, c
    assert any("쌍꺼풀" in n and "기준" in n for n in f.notes)


def test_crease_reference_can_be_chosen(hub, creased):
    for ref in ("왼눈", "오른눈"):
        c = _render(hub, creased, {"crease"}, crease_ref=ref).outputs["crease"]
        assert c["ref"] == ref and c["after_diff"] <= 2.5, (ref, c)


# ------------------------------------------------------------ 콧볼 옆 그림자

def test_nose_side_shadow_lifted(hub, portrait):
    a = analyze("", hub, rgb=portrait)
    p = a.points
    fw = R.face_w(p)
    nose_c = (p["왼콧볼"][0] + p["오른콧볼"][0]) / 2
    wx, wy = p["왼콧볼"]
    cx, cy = wx - 0.075 * fw, wy + 0.01 * fw
    m = np.zeros(portrait.shape[:2], np.float32)
    cv2.ellipse(m, (round(cx), round(cy)), (round(0.035 * fw), round(0.04 * fw)), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 0.01 * fw)[..., None]
    shadowed = np.clip(portrait * (1 - 0.22 * m), 0, 255).astype(np.uint8)
    on = _render(hub, shadowed, {"nose_shadow"})
    off = _render(hub, shadowed, set())
    L = lambda im: float(cv2.cvtColor(im[round(cy) - 2:round(cy) + 3, round(cx) - 2:round(cx) + 3],
                                      cv2.COLOR_RGB2LAB)[..., 0].mean())
    assert L(on.outputs["retouched"]) > L(off.outputs["retouched"]) + 5
    assert any("왼쪽" in d for d in on.outputs["nose_shadow"])
    # 콧볼에서 먼 곳(눈, 이마)은 그대로
    ex, ey = (round(v) for v in p["왼눈_홍채중심"])
    assert np.array_equal(on.outputs["retouched"][ey - 5:ey + 5, ex - 5:ex + 5],
                          off.outputs["retouched"][ey - 5:ey + 5, ex - 5:ex + 5])
    assert nose_c > wx
