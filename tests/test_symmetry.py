"""2단계: 얼굴 대칭 (가이드 §9)."""
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
from photostory.core.warp import Warp

OTHERS = {"skin", "tone", "sharpen", "catchlight"}


@pytest.fixture(scope="module")
def crooked(hub, portrait):
    """코·입을 오른쪽으로 12px, 사진 오른쪽 눈을 위로 4px 옮긴 사진."""
    a = analyze("", hub, rgb=portrait)
    fw = R.face_w(a.points)
    w = Warp(portrait.shape, R.face_box(a.regions, portrait.shape, 0.4 * fw))

    def piece(keys, grow):
        m = np.zeros(w.roi_shape, np.uint8)
        for k in keys:
            m |= R.poly_mask(w.roi_shape, w.to_local(a.regions[k]), hull=True)
        return R.feather(R.dilate(m, grow), 0.05 * fw)

    w.shift(piece(("코아래", "입술"), 0.05 * fw), 12, 0)
    w.shift(piece(("오른눈",), 0.06 * fw), 0, -4)
    return w.apply_image(portrait)


def _run(hub, img, strength=None, preview=False):
    path = Path(tempfile.mkdtemp()) / "a.png"
    cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(str(path))
    s = Session(Customer(name="a", purposes=["passport"]), original=path)
    s.disabled_steps = set(OTHERS)
    if strength is not None:
        for k in ("jaw", "nose", "mouth", "eyes"):
            s.customer.strength.set(k, strength)
    proc = Processor(hub, SpecRegistry())
    proc.start(s)
    f = proc.render(s, preview=preview).frame
    proc.clear()
    return f


def test_symmetry_fixes_nose_mouth_eye(hub, crooked):
    f = _run(hub, crooked)
    sym = f.outputs["symmetry"]
    b, a = sym["before"], sym["after"]
    # MediaPipe 얼굴 점은 평균 얼굴 쪽으로 끌려서 12px 옮긴 코·입을 절반 정도로만 잰다 (진행상황.md 2-2)
    assert b["코"] > 4 and b["입"] > 4, b
    # 코·입은 한 조각으로 평균만큼 옮기므로 둘의 차이는 남을 수 있다 (가이드도 같은 방식)
    assert abs(a["코"] + a["입"]) / 2 < 1.0 and abs(a["코"]) < 2.5 and abs(a["입"]) < 2.5, a
    assert any(m.startswith("코·입") for m in sym["moved"]) and any("높이" in m for m in sym["moved"])
    for key in ("볼라인", "눈동자"):
        if key in b:
            assert abs(a[key]) <= abs(b[key]) + 1.0, (key, b, a)    # 다른 곳을 나쁘게 만들지 않음


def test_symmetry_strength(hub, crooked):
    f0 = _run(hub, crooked, 0.0)
    assert f0.outputs["symmetry"]["moved"] == []
    half = _run(hub, crooked, 0.5).outputs["symmetry"]
    b, a = half["before"]["코"], half["after"]["코"]
    assert 0.3 * b < a < 0.75 * b, (b, a)                           # 반쯤만
    strong = _run(hub, crooked, 1.4).outputs["symmetry"]
    assert strong["after"]["코"] > -2.5                             # 강하게도 반대쪽으로 넘지 않음


def test_symmetry_leaves_symmetric_face_alone(hub, portrait):
    """왼쪽 절반을 뒤집어 붙인 완전 대칭 얼굴: 코·입은 5px 기준 아래라 옮기지 않는다."""
    a = analyze("", hub, rgb=portrait)
    p = a.points
    cx = round((p["왼눈_안쪽꼭짓점"][0] + p["오른눈_안쪽꼭짓점"][0]) / 2)
    sym = portrait.copy()
    w = min(cx, sym.shape[1] - cx)
    sym[:, cx:cx + w] = portrait[:, cx - w:cx][:, ::-1]
    # 사진관 원본 크기로 키워서 본다: 5px 기준은 원본(얼굴 폭 약 1200px) 기준이고,
    # 테스트 사진 크기(얼굴 폭 260px)에서는 1px 남짓이라 측정 흔들림보다 작다
    big = cv2.resize(sym, None, fx=2.9116, fy=2.9116, interpolation=cv2.INTER_CUBIC)
    f = _run(hub, big, preview=True)
    out = f.outputs["symmetry"]
    to_orig = 1 / f.scale
    assert not any(m.startswith("코·입") for m in out["moved"]), out
    assert abs(out["before"]["코"]) * to_orig < 5 and abs(out["before"]["입"]) * to_orig < 5, out
