"""1단계 피부·명암 (규칙 방식) 테스트."""
import cv2
import numpy as np
import pytest

from photostory.core.measure import regions as R
from photostory.core.prepare import analyze
from photostory.core.processor import Processor
from photostory.core.session import Customer, Session
from photostory.core.specs import SpecRegistry


def _cheeks(a):
    """사진 왼쪽·오른쪽 볼 가운데 (콧볼과 입 높이 볼 사이, 조금 위)."""
    p = a.points
    out = []
    for nose, cheek, eye in (("왼콧볼", "왼볼_입높이", "왼눈_바깥꼭짓점"),
                             ("오른콧볼", "오른볼_입높이", "오른눈_바깥꼭짓점")):
        x = (p[nose][0] + p[cheek][0]) / 2
        y = (p[nose][1] + p[eye][1]) / 2 + 0.25 * (p[nose][1] - p[eye][1])
        out.append((round(x), round(y)))
    return out


@pytest.fixture(scope="module")
def spotted(hub, portrait):
    """볼에 갈색 점(남겨야 함)과 붉은 트러블(지워야 함)을 하나씩 그린 사진."""
    a = analyze("", hub, rgb=portrait)
    mm = R.face_w(a.points) / 140.0
    (lx, ly), (rx, ry) = _cheeks(a)
    img = portrait.copy()
    cv2.circle(img, (lx, ly), max(2, round(1.3 * mm)), (95, 62, 45), -1, cv2.LINE_AA)
    pim = np.zeros(img.shape[:2], np.float32)
    cv2.circle(pim, (rx, ry), max(2, round(1.5 * mm)), 1.0, -1, cv2.LINE_AA)
    pim = cv2.GaussianBlur(pim, (0, 0), 0.6 * mm)[..., None] / max(pim.max(), 1e-6)
    pim = np.clip(pim / pim.max(), 0, 1)
    shift = np.array([8, -38, -30], np.float32)       # 붉고 조금 어둡게
    img = np.clip(img.astype(np.float32) + pim * shift, 0, 255).astype(np.uint8)
    return img, (lx, ly), (rx, ry)


def test_spot_kinds(hub, spotted):
    img, mole, pimple = spotted
    a = analyze("", hub, rgb=img)
    near = lambda pt: [s for s in a.extra["spots"] if np.hypot(s.x - pt[0], s.y - pt[1]) < 6]
    m, p = near(mole), near(pimple)
    assert m and m[0].keep, a.extra["spots"]
    assert p and p[0].kind == "트러블", a.extra["spots"]
    assert len(a.regions["보존점"]) == sum(s.keep for s in a.extra["spots"])


def _session(tmp_path, img, name="홍길동"):
    path = tmp_path / f"{name}.png"
    cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(str(path))
    return Session(Customer(name=name, purposes=["passport"]), original=path)


def _lum(img, pt, r=2):
    x, y = pt
    return float(cv2.cvtColor(img[y - r:y + r + 1, x - r:x + r + 1], cv2.COLOR_RGB2LAB)[..., 0].mean())


def test_skin_keeps_mole_removes_trouble(hub, spotted, tmp_path):
    img, mole, pimple = spotted
    proc = Processor(hub, SpecRegistry())
    s = _session(tmp_path, img)
    s.disabled_steps = {"level", "tone"}              # 좌표가 바뀌지 않게 수평은 끈다
    proc.start(s)
    out = proc.render(s, preview=False).frame
    ret = out.outputs["retouched"]
    # 점: 거의 그대로 / 트러블: 주변 피부 쪽으로 많이 옅어짐
    assert abs(_lum(ret, mole) - _lum(img, mole)) < 6
    ring = lambda im, pt: float(np.median(cv2.cvtColor(im[pt[1] - 12:pt[1] + 13, pt[0] - 12:pt[0] + 13],
                                                       cv2.COLOR_RGB2LAB)[..., 0]))
    before = ring(img, pimple) - _lum(img, pimple)
    after = ring(ret, pimple) - _lum(ret, pimple)
    assert after < 0.6 * before, (before, after)
    assert any("점" in n and "남김" in n for n in out.notes)
    # 눈은 건드리지 않음
    ex, ey = (round(v) for v in out.points["왼눈_홍채중심"])
    assert np.abs(ret[ey - 2:ey + 3, ex - 2:ex + 3].astype(int) - img[ey - 2:ey + 3, ex - 2:ex + 3]).max() <= 1
    proc.clear()


def test_strength_zero_and_isolation(hub, portrait, tmp_path):
    proc = Processor(hub, SpecRegistry())
    s = _session(tmp_path, portrait)
    proc.start(s)
    s.disabled_steps = {"skin", "tone"}
    plain = proc.render(s, preview=True).frame.outputs["retouched"]
    s.disabled_steps = set()
    for k in ("skin", "tone"):
        s.customer.strength.set(k, 0.0)
    zero = proc.render(s, preview=True).frame.outputs["retouched"]
    assert np.array_equal(plain, zero)
    s.customer.strength.set("skin", 1.0)
    s.customer.strength.set("tone", 1.0)
    on = proc.render(s, preview=True).frame.outputs["retouched"]
    assert not np.array_equal(plain, on)
    # 명암만 바꾸면 피부 단계는 다시 계산하지 않는다
    s.customer.strength.set("tone", 1.4)
    r = proc.render(s, preview=True)
    assert "skin" not in r.timings and "tone" in r.timings
    # 눈을 바꾸면 눈 반사점(5)부터 다시 계산한다. 그 앞의 피부(3·4)는 다시 하지 않는다
    s.customer.strength.set("eyes", 1.4)
    r = proc.render(s, preview=True)
    assert "skin" not in r.timings and "catchlight" in r.timings
    proc.clear()


def test_tone_brightens_bridge_darkens_jaw(hub, portrait, tmp_path):
    proc = Processor(hub, SpecRegistry())
    s = _session(tmp_path, portrait)
    s.disabled_steps = {"level", "skin"}
    proc.start(s)
    on = proc.render(s, preview=False).frame
    s.disabled_steps = {"level", "skin", "tone"}
    off = proc.render(s, preview=False).frame
    a, b = on.outputs["retouched"], off.outputs["retouched"]
    bridge = on.regions["콧대"][2]
    assert _lum(a, tuple(round(v) for v in bridge)) > _lum(b, tuple(round(v) for v in bridge)) + 1
    jaw = on.regions["왼턱선_선"][4]
    jx, jy = round(jaw[0]) + 3, round(jaw[1]) - 3                  # 턱선 조금 안쪽
    assert _lum(a, (jx, jy)) < _lum(b, (jx, jy)) - 0.5
    proc.clear()
