import cv2
import numpy as np
import pytest

from photostory.core.export import Conflict, planned_names, save_all
from photostory.core.io.imageio import read_image
from photostory.core.processor import Processor
from photostory.core.session import Customer, Session
from photostory.core.specs import A4_PX, SpecRegistry
from photostory.steps.a4 import brightness_lut, build_a4


@pytest.fixture
def proc(hub):
    p = Processor(hub, SpecRegistry())
    yield p
    p.clear()


def _session(tmp_path, img, purposes, name="홍길동"):
    path = tmp_path / f"{name}_원본.png"
    cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tofile(str(path))
    return Session(Customer(name=name, purposes=purposes), original=path)


def test_end_to_end_single_spec(proc, portrait, tmp_path):
    s = _session(tmp_path, portrait, ["passport"])
    proc.start(s)
    r = proc.render(s, preview=False)
    out = r.frame.outputs
    t = out["transfers"]["passport"]
    assert t.shape == (531, 413, 3)
    chk = out["transfer_checks"]["passport"]
    assert chk.ok, chk
    assert abs(chk.crown - 15.4) < 1 and 405 <= chk.chin <= 418
    assert out["retouched"].shape == portrait.shape
    assert out["a4"].shape == (A4_PX[1], A4_PX[0], 3)
    files = save_all(tmp_path / "out", "홍길동", proc.chosen_specs(s), out)
    assert [f.name for f in files] == ["홍길동_수정본.jpg", "홍길동_전송.jpg", "홍길동_a4출력.jpg"]
    assert read_image(files[1]).dpi == (300.0, 300.0)
    # 같은 이름: 번호 붙이기
    again = save_all(tmp_path / "out", "홍길동", proc.chosen_specs(s), out, Conflict.NUMBER)
    assert again[0].name == "홍길동_수정본(2).jpg"
    assert save_all(tmp_path / "out", "홍길동", proc.chosen_specs(s), out, Conflict.CANCEL) is None


def test_multi_spec_and_color_background(proc, portrait, tmp_path):
    s = _session(tmp_path, portrait, ["passport", "visa_my"])
    proc.start(s)
    out = proc.render(s, preview=True).frame.outputs
    my = out["transfers"]["visa_my"]
    assert my.shape == (591, 413, 3)
    assert np.abs(my[2, 2].astype(int) - (0x13, 0x99, 0xF8)).max() <= 2
    assert (out["transfers"]["passport"][2, 2] == 255).all()
    names = [n for _, n in planned_names("홍길동", proc.chosen_specs(s))]
    assert "홍길동_전송_여권.jpg" in names and "홍길동_전송_말레이시아비자.jpg" in names


def test_level_recovers_tilt(proc, portrait, tmp_path):
    h, w = portrait.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 3.0, 1.0)
    tilted = cv2.warpAffine(portrait, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    s0 = _session(tmp_path, portrait, ["half"], "a")
    s1 = _session(tmp_path, tilted, ["half"], "b")
    proc.start(s0)
    base = proc.render(s0, preview=True).frame.outputs["tilt_deg"]
    proc.clear()
    proc.start(s1)
    tilt = proc.render(s1, preview=True).frame.outputs["tilt_deg"]
    assert abs((tilt - base) - (-3.0)) < 0.5


def test_strength_change_reuses_cache(proc, portrait, tmp_path):
    s = _session(tmp_path, portrait, ["half"])
    proc.start(s)
    proc.render(s, preview=True)
    s.placement("half").scale = 0.05
    r = proc.render(s, preview=True)
    assert "level" not in r.timings and "transfer" in r.timings


def test_a4_layout_rules():
    reg = SpecRegistry()
    pp, half = reg.get("passport"), reg.get("half")
    tr = {"passport": np.full((531, 413, 3), 100, np.uint8), "half": np.full((472, 354, 3), 100, np.uint8)}
    a4, notes = build_a4("윤지수", [pp, half], tr, None)
    lut = brightness_lut(5)
    assert a4[22, 146].tolist() == [lut[100]] * 3                 # 첫 사진 x=146, y=22
    assert a4[22, 146 + 413 + 30].tolist() == [lut[100]] * 3      # 간격 30
    assert a4[21, 300].tolist() == [lut[180]] * 3                 # 바깥쪽 1px 획
    y2 = 22 + 531 + 35
    assert a4[y2, 146].tolist() == [lut[100]] * 3                 # 두 번째 줄 y=588
    assert a4[y2, 146 + 5 * (354 + 30)].tolist() == [lut[100]] * 3  # 반명함판 6장
    assert (a4[3000, 1000] == 255).all()
    assert any("글꼴" in n for n in notes)


def test_a4_label_drawn():
    from photostory.steps.a4 import DEFAULT_FONT
    reg = SpecRegistry()
    tr = {"passport": np.full((531, 413, 3), 100, np.uint8)}
    a4, notes = build_a4("윤지수", [reg.get("passport")], tr, DEFAULT_FONT)
    region = a4[21:553, 30:110]
    assert (region < 250).any() and not notes
