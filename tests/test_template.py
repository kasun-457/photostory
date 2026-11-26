"""측정 안내선 템플릿 (가이드 §5)."""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from photostory.core.measure.template import _ears, _jaw_neck_bend, _pair, _row_edges, _shadow_jaw
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
            assert t.get("8") is not None or any("귀" in n for n in t.notes)
    proc.clear()


def _oval(cx=400, cy=300, a=150, b=200):
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    return np.stack([cx + a * np.cos(t), cy + b * np.sin(t)], axis=1).astype(np.float32)


def test_ears_found_outside_face():
    """얼굴 윤곽 바깥에 붙은 피부 덩어리(귀): 아래 끝과 바깥 끝."""
    skin = np.zeros((800, 800), np.float32)
    cv2.ellipse(skin, (400, 300), (155, 205), 0, 0, 360, 1.0, -1)
    cv2.ellipse(skin, (245, 300), (28, 60), 0, 0, 360, 1.0, -1)     # 왼쪽 귀: 아래 끝 360, 바깥 217
    cv2.ellipse(skin, (557, 290), (25, 55), 0, 0, 360, 1.0, -1)     # 오른쪽 귀: 아래 끝 345, 바깥 582
    l, r = _ears(skin, _oval(), 180, 420, 300)
    assert l is not None and r is not None
    assert abs(l[0] - 360) <= 2 and abs(l[1] - 217) <= 2
    assert abs(r[0] - 345) <= 2 and abs(r[1] - 582) <= 2
    plain = np.zeros((800, 800), np.float32)
    cv2.ellipse(plain, (400, 300), (155, 205), 0, 0, 360, 1.0, -1)
    assert _ears(plain, _oval(), 180, 420, 300) == (None, None)    # 귀가 안 보이면 없음


def test_shadow_jaw_upper_edge():
    """얼굴살: 바깥 윤곽은 곧은 기둥인데, 안쪽에 턱 그림자가 있다. 그림자 '위쪽 경계'가 목 옆선과 만나는 점."""
    h, w = 900, 800
    union = np.zeros((h, w), np.float32)
    union[100:, 250:551] = 1.0                                      # 볼~목이 한 줄
    gray = np.full((h, w), 190, np.float32)
    gray[:, :250] = gray[:, 551:] = 240
    xs = np.arange(w)
    top = 520 + 0.0012 * (xs - 400) ** 2                          # 그림자 위 경계: 가운데 520, 옆 547
    for x in range(250, 551):
        gray[int(top[x]):int(top[x]) + 60, x] = 120               # 그림자 띠 (아래로 60px)
    b = _shadow_jaw(gray, union, 400, 420, 600, 300)
    assert b is not None
    xl, xr, y = b
    assert abs(xl - 259) <= 3 and abs(xr - 541) <= 3
    exp = 520 + 0.0012 * (259 - 400) ** 2
    assert abs(y - exp) <= 4, (y, exp)                             # 어두운 가운데(+30)가 아니라 위 경계
