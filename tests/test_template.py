"""측정 안내선 템플릿 (가이드 §5)."""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from photostory.core.measure.template import _jaw_neck_bend, _pair, _row_edges
from photostory.core.processor import Processor
from photostory.core.session import Customer, Gender, Session
from photostory.core.specs import SpecRegistry


def _face_neck(neck_l=320, neck_r=480):
    m = np.zeros((800, 800), np.float32)
    cv2.ellipse(m, (400, 300), (150, 200), 0, 0, 360, 1.0, -1)
    m[380:, neck_l:neck_r + 1] = 1.0
    return m


def _bend_y(x):
    """타원 (400,300) 반지름 (150,200)이 세로선 x와 만나는 아래쪽 y."""
    return 300 + 200 * np.sqrt(1 - ((x - 400) / 150) ** 2)


def test_jaw_neck_bend_symmetric():
    b = _jaw_neck_bend(_face_neck(), 400, 300, 640, 300, 500, 300)
    assert b is not None
    l, r, y = b
    assert abs(l - 320) <= 4 and abs(r - 480) <= 4
    assert abs(y - _bend_y(320)) <= 8


def test_jaw_neck_bend_takes_lower_side():
    b = _jaw_neck_bend(_face_neck(neck_l=340), 400, 300, 640, 300, 500, 300)
    assert b is not None
    l, r, y = b
    assert abs(l - 340) <= 4 and abs(r - 480) <= 4
    assert abs(y - max(_bend_y(340), _bend_y(480))) <= 8     # 낮은 쪽 (y가 큰 쪽)


def test_no_neck_gives_none():
    m = np.zeros((800, 800), np.float32)
    cv2.ellipse(m, (400, 300), (150, 200), 0, 0, 360, 1.0, -1)
    assert _jaw_neck_bend(m, 400, 300, 640, 300, 500, 300) is None


def test_row_edges_starts_outside():
    m = _face_neck()
    l, r = _row_edges(m, 460, 200, 600, 400)      # 시작점이 마스크 밖이어도 안쪽에서 찾음
    assert l is not None and r is not None and l < 400 < r


def test_pair_uses_narrow_side():
    d, sides = _pair(400, 300, 520)
    assert d == 100 and sides == (300, 520)


def test_template_lines(hub, portrait):
    path = Path(tempfile.mkdtemp()) / "a.png"
    cv2.imencode(".png", cv2.cvtColor(portrait, cv2.COLOR_RGB2BGR))[1].tofile(str(path))
    proc = Processor(hub, SpecRegistry())
    for g, order in ((Gender.FEMALE, "ABCDEFGHJ"), (Gender.MALE, ["1", "2", "3", "4", "5", "6", "7", "9", "11"])):
        s = Session(Customer(name="a", purposes=["passport"], gender=g), original=path)
        proc.start(s)
        t = proc.render(s, preview=True).frame.outputs["template"]
        ys = [t.get(i).pos for i in order]
        assert ys == sorted(ys), (g, ys)                       # 위에서 아래 순서
        for line in t.v:
            if line.pair:
                a, b = t.v_x(line)
                assert abs((a + b) / 2 - t.center_x) < 1e-6     # 중심선 대칭
                l, r = line.sides
                assert line.pos <= min(t.center_x - l, r - t.center_x) + 1e-6   # 좁은 쪽
        if g == Gender.MALE:
            assert t.get("8") is None and any("귀" in n for n in t.notes)
    proc.clear()
