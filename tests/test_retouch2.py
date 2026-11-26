"""1단계: 선명하게(수정-02, 흐릴 때만), 눈 반사점 (반대쪽 눈 따오기)."""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from photostory.core.measure.catchlight import glare_fraction, iris_circle, judge
from photostory.core.measure.sharpness import SOFT, feature_sharpness
from photostory.core.prepare import analyze
from photostory.core.processor import Processor
from photostory.core.session import Customer, Session
from photostory.core.specs import SpecRegistry


def _session(img, name="a"):
    path = Path(tempfile.mkdtemp()) / f"{name}.png"
    cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(str(path))
    return Session(Customer(name=name, purposes=["passport"]), original=path)


# ------------------------------------------------------------ 선명하게

def test_sharpness_same_across_resolution(hub, portrait):
    a = analyze("", hub, rgb=portrait)
    s1 = feature_sharpness(portrait, a.points, a.regions)
    big = cv2.resize(portrait, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    b = analyze("", hub, rgb=big)
    s2 = feature_sharpness(big, b.points, b.regions)
    assert abs(s1 - s2) < 0.015, (s1, s2)
    blurred = cv2.GaussianBlur(portrait, (0, 0), 2)
    c = analyze("", hub, rgb=blurred)
    assert feature_sharpness(blurred, c.points, c.regions) < SOFT <= s1


def test_sharpen_only_when_soft(hub, portrait):
    proc = Processor(hub, SpecRegistry())
    # 또렷한 편 → 하지 않음 (메모만)
    s = _session(portrait)
    s.disabled_steps = {"skin", "tone", "catchlight"}
    proc.start(s)
    f = proc.render(s, preview=False).frame
    assert any("선명하게: 하지 않음" in n for n in f.notes)
    proc.clear()
    # 흐림 → 함, 눈 둘레가 또렷해짐
    soft = cv2.GaussianBlur(portrait, (0, 0), 2)
    s = _session(soft, "b")
    s.disabled_steps = {"skin", "tone", "catchlight", "level", "symmetry"}
    proc.start(s)
    on = proc.render(s, preview=False).frame
    assert any("선명하게: 함" in n for n in on.notes)
    a = proc.pipeline.analysis
    after = feature_sharpness(on.outputs["retouched"], a.points, a.regions)
    assert after > a.extra["sharpness"] * 1.05      # 반경 1.8px(원본 기준)이 작은 얼굴에선 0.5px라 효과가 작음
    # 볼(피부)은 그대로
    cx, cy = (round(v) for v in a.points["왼볼_입높이"])
    cx += round(0.25 * (a.points["코끝"][0] - cx))
    assert np.abs(on.outputs["retouched"][cy - 3:cy + 4, cx - 3:cx + 4].astype(int)
                  - soft[cy - 3:cy + 4, cx - 3:cx + 4]).max() <= 2
    proc.clear()


# ------------------------------------------------------------ 눈 반사점

@pytest.fixture(scope="module")
def glared(hub, portrait):
    """사진 오른쪽 눈 동공을 큰 반사로 덮은 사진."""
    a = analyze("", hub, rgb=portrait)
    (c, r) = iris_circle(a.points, "오른눈")
    img = portrait.copy()
    cv2.circle(img, (round(c[0]), round(c[1])), max(2, round(0.5 * r)), (250, 250, 248), -1, cv2.LINE_AA)
    return img, c, r


def test_glare_judged(hub, portrait, glared):
    a = analyze("", hub, rgb=portrait)
    assert judge(portrait, a.points)["fix"] is None          # 원래 사진은 고치지 않음
    img, c, r = glared
    b = analyze("", hub, rgb=img)
    j = judge(img, b.points)
    assert j["fix"] == "오른눈", j
    assert glare_fraction(img, c, r) > 0.15


def test_catchlight_fixed(hub, portrait, glared):
    img, c, r = glared
    proc = Processor(hub, SpecRegistry())
    s = _session(img, "c")
    s.disabled_steps = {"skin", "tone", "sharpen", "level", "symmetry"}
    proc.start(s)
    f = proc.render(s, preview=False).frame
    out = f.outputs["retouched"]
    assert any("눈 반사점" in n for n in f.notes)
    assert glare_fraction(out, c, r) < 0.06                   # 뚫린 반사가 사라짐
    # 홍채 바깥(흰자·피부)은 거의 그대로
    x, y = round(c[0] + 2.2 * r), round(c[1] + 2.2 * r)
    assert np.abs(out[y - 2:y + 3, x - 2:x + 3].astype(int) - img[y - 2:y + 3, x - 2:x + 3]).max() <= 1
    # 강도(눈) 0이면 하지 않음
    s.customer.strength.set("eyes", 0.0)
    f0 = proc.render(s, preview=False).frame
    assert glare_fraction(f0.outputs["retouched"], c, r) > 0.15
    proc.clear()
