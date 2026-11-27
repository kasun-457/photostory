"""3-3 정답 지도 뽑기: 알려진 유동화·톤·피부 비율을 만들어 넣고 되찾는지 (학습설계 3)."""
import cv2
import numpy as np

from training import extract_targets as ex

H, W = 256, 224


def _texture(seed=0):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.uniform(60, 200, (H, W, 3)).astype(np.float32), (0, 0), 6)
    grain = rng.normal(0, 14, (H, W, 3)).astype(np.float32)          # 피부 결 같은 잔무늬
    return np.clip(base + grain, 0, 255).astype(np.uint8)


def _masks():
    person = np.zeros((H, W), np.uint8)
    cv2.ellipse(person, (W // 2, H // 2), (80, 100), 0, 0, 360, 255, -1)
    valid = np.full((H, W), 255, np.uint8)
    return person, person.copy(), valid


class FakeFlow:
    """정해 둔 흐름(조각 px)을 1/2 해상도로 돌려준다."""

    def __init__(self, flow):
        self.flow = flow

    def __call__(self, a, b, iters=12):
        h, w = a.shape[:2]
        return cv2.resize(self.flow, (w, h), interpolation=cv2.INTER_AREA) / ex.FLOW_DOWN


def _smooth_flow(dx):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bump = np.exp(-(((xx - W / 2) / 40) ** 2 + ((yy - H / 2) / 50) ** 2))
    return np.dstack([dx * bump, np.zeros_like(bump)]).astype(np.float32)


def test_identity_pair_gives_empty_maps():
    o = _texture()
    person, skin, valid = _masks()
    maps, st = ex.extract(o, o.copy(), person, skin, valid, FakeFlow(np.zeros((H, W, 2), np.float32)))
    assert st["err_raw"] == 0 and st["err_skin"] < 0.5
    assert np.abs(maps["gain"].astype(np.float32)).max() < 0.01
    assert maps["alpha"].mean() < 3
    assert maps["fill"].max() == 0


def test_flow_and_gain_are_explained():
    o = _texture(1)
    person, skin, valid = _masks()
    flow = _smooth_flow(6.0)
    r = ex.remap(o.astype(np.float32), flow) * np.float32(0.9)        # 옮기고 10% 어둡게
    r = np.clip(r, 0, 255).astype(np.uint8)
    maps, st = ex.extract(o, r, person, skin, valid, FakeFlow(flow))
    assert st["err_flow"] < st["err_raw"]
    assert st["err_tone"] < 1.5
    g = np.exp(maps["gain"].astype(np.float32)).mean(2)
    inner = cv2.resize(cv2.erode(person, np.ones((31, 31), np.uint8)), (g.shape[1], g.shape[0])) > 128
    assert abs(float(np.median(g[inner])) - 0.9) < 0.02
    assert st["fill_frac"] < 0.01


def test_skin_ratio_is_recovered():
    o = _texture(2)
    person, skin, valid = _masks()
    s1 = ex.snapshot1_clean(o.astype(np.float32))
    r = np.clip(o + 0.6 * (s1 - o), 0, 255).astype(np.uint8)
    maps, st = ex.extract(o, r, person, skin, valid, FakeFlow(np.zeros((H, W, 2), np.float32)))
    a = maps["alpha"].astype(np.float32) / 255
    inner = cv2.resize(cv2.erode(skin, np.ones((31, 31), np.uint8)), (a.shape[1], a.shape[0])) > 128
    assert abs(float(np.median(a[inner])) - 0.6) < 0.1
    assert st["err_skin"] < st["err_tone"]


def test_unexplained_patch_becomes_fill():
    o = _texture(3)
    person, skin, valid = _masks()
    r = o.copy()
    cv2.circle(r, (W // 2 + 20, H // 2), 8, (230, 40, 40), -1)        # 무늬로 설명 안 되는 것
    maps, _ = ex.extract(o, r, person, skin, valid, FakeFlow(np.zeros((H, W, 2), np.float32)))
    assert maps["fill"][H // 2, W // 2 + 20] == 255
    assert maps["fill"][H // 2, W // 2 - 40] == 0


def test_whole_body_shift_goes_to_affine_not_liquify():
    o = _texture(4)
    person, skin, valid = _masks()
    flow = np.zeros((H, W, 2), np.float32)
    flow[..., 0], flow[..., 1] = 5.0, -3.0                             # 몸 전체를 옮긴 것 (배치)
    r = np.clip(ex.remap(o.astype(np.float32), flow), 0, 255).astype(np.uint8)
    maps, st = ex.extract(o, r, person, skin, valid, FakeFlow(flow))
    assert abs(st["g_dx"] - 5.0) < 0.3 and abs(st["g_dy"] + 3.0) < 0.3
    assert abs(st["g_scale"] - 1.0) < 0.01
    assert st["flow_max"] < 0.5                                       # 유동화 지도에는 남지 않음
    assert st["err_flow"] < st["err_raw"] / 3
