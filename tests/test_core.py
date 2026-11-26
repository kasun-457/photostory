import numpy as np
import pytest

from photostory.core.io.imageio import read_image, write_jpeg
from photostory.core.pipeline import Analysis, Pipeline, RenderParams, Step
from photostory.core.session import Strength
from photostory.core.specs import SpecRegistry


def test_builtin_specs_match_guide_table():
    reg = SpecRegistry()
    specs = reg.all()
    assert len(specs) == 14
    half = reg.get("half")
    assert half.size_px == (354, 472)
    assert half.crown_line == pytest.approx(23.6)
    assert half.chin_min_line == pytest.approx(301.2)
    assert half.chin_max_line == pytest.approx(313.0)
    pp = reg.get("passport")
    assert (pp.crown_line, pp.chin_min_line, pp.chin_max_line) == (15.4, 405.6, 417.4)
    assert reg.get("visa_us").size_px == (600, 600)
    assert reg.get("visa_ca").chin_min_line == pytest.approx(496.1)
    assert reg.get("visa_my").bg_rgb == (0x13, 0x99, 0xF8)
    assert reg.find("면허증").id == "passport"
    assert reg.find("이력서 사진").id == "half"


def test_spec_dpi_and_a4_formula():
    reg = SpecRegistry()
    s = reg.get("half")
    s2 = type(s)(**{**s.to_json(), "dpi": 640, "a4_count": None})
    assert s2.size_px == (756, 1008)
    assert s2.a4_count_resolved() == 6
    assert s2.crown_line == pytest.approx(23.6 * 640 / 300)


def test_spec_validate_and_user_save(tmp_path):
    reg = SpecRegistry(tmp_path)
    s = reg.get("half")
    bad = type(s)(**{**s.to_json(), "id": "x", "top_cm": 2.0, "head_cm": 2.5})
    assert bad.validate()
    good = type(s)(**{**s.to_json(), "id": "custom1", "name": "사용자 규격"})
    reg.save(good)
    assert reg.get("custom1").name == "사용자 규격"
    reg.delete("half")
    assert all(x.id != "half" for x in reg.all())
    reg.reset_builtin("half")
    assert any(x.id == "half" for x in reg.all())


def test_jpeg_roundtrip_korean_path(tmp_path):
    img = np.zeros((40, 30, 3), np.uint8)
    img[:, :, 0] = 200
    p = write_jpeg(tmp_path / "홍길동_수정본.jpg", img, dpi=300)
    li = read_image(p)
    assert li.rgb.shape == (40, 30, 3)
    assert li.dpi == (300.0, 300.0)
    assert abs(int(li.rgb[..., 0].mean()) - 200) <= 2


def test_strength_levels():
    s = Strength.from_level("강하게")
    assert s.get("skin") == 1.4
    s.set("eyes", 0.5)
    assert s.level is None
    s.set_all("약하게")
    assert s.level == "약하게" and s.get("eyes") == 0.6


class _Count(Step):
    def __init__(self, sid, keys, cache=False):
        self.id, self.title, self.param_keys, self.cache = sid, sid, keys, cache
        self.calls = 0

    def run(self, frame, analysis, params):
        self.calls += 1
        frame.outputs[self.id] = params.get(self.param_keys[0]) if self.param_keys else True
        return frame


def test_pipeline_reruns_only_changed_steps():
    a, b, c = _Count("a", (), cache=True), _Count("b", ("x",)), _Count("c", ("y",))
    pipe = Pipeline([a, b, c])
    pipe.reset(Analysis(np.zeros((10, 10, 3), np.uint8), {}))
    pipe.render(RenderParams({"x": 1, "y": 1}), preview=True)
    assert (a.calls, b.calls, c.calls) == (1, 1, 1)
    pipe.render(RenderParams({"x": 1, "y": 1}), preview=True)
    assert (a.calls, b.calls, c.calls) == (1, 1, 1)
    r = pipe.render(RenderParams({"x": 1, "y": 2}), preview=True)
    assert (a.calls, b.calls, c.calls) == (1, 2, 2)  # a 결과는 캐시에서
    assert r.frame.outputs == {"a": True, "b": 1, "c": 2}
