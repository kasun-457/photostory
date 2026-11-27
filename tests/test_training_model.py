"""3-4 보정 지도 모델: 지도 적용 식, 좌우 뒤집기, 모델 출력 모양 (학습설계 3)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training import extract_targets as ex  # noqa: E402
from training.dataset import N_AUX, flip_affine  # noqa: E402
from training.model import CorrectionMapNet, apply_maps  # noqa: E402

H, W = 48, 40


def _img(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 0.9, (H, W, 3)).astype(np.float32)


def _t(a):
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))[None]


def test_zero_maps_return_original():
    o = _t(_img())
    z2, z3, z1 = torch.zeros(1, 2, H, W), torch.zeros(1, 3, H, W), torch.zeros(1, 1, H, W)
    out = apply_maps(o, o, z2, None, z3, z1)
    assert torch.allclose(out, o, atol=1e-5)


def test_flow_matches_extraction_remap():
    o = _img(1)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    flow = np.dstack([np.sin(yy / 7) * 2.5, np.cos(xx / 5) * 1.5]).astype(np.float32)
    want = ex.remap(o * 255, flow) / 255
    got = apply_maps(_t(o), _t(o), _t(flow), None, torch.zeros(1, 3, H, W), torch.zeros(1, 1, H, W))
    inner = (slice(4, -4), slice(4, -4))
    assert np.abs(got[0].permute(1, 2, 0).numpy()[inner] - want[inner]).max() < 1e-3


def test_affine_then_strength_and_skin():
    o, s1 = _t(_img(2)), _t(_img(3))
    A = torch.tensor([[[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]]])      # 2px 오른쪽에서 가져옴
    z2, z3 = torch.zeros(1, 2, H, W), torch.zeros(1, 3, H, W)
    shifted = apply_maps(o, s1, z2, A, z3, torch.zeros(1, 1, H, W))
    assert torch.allclose(shifted[..., :, :-2], o[..., :, 2:], atol=1e-5)
    half = apply_maps(o, s1, z2, None, z3, torch.full((1, 1, H, W), 0.5))
    assert torch.allclose(half, 0.5 * o + 0.5 * s1, atol=1e-5)
    zero = apply_maps(o, s1, z2, None, torch.full((1, 3, H, W), 0.3), torch.full((1, 1, H, W), 0.5), strength=0.0)
    assert torch.allclose(zero, o, atol=1e-5)                    # 강도 0 = 그대로


def test_flip_affine_is_consistent():
    o = _img(4)
    A = np.array([[0.99, -0.03, 1.5], [0.03, 0.99, -2.0]])
    def warp(img, A):
        return apply_maps(_t(img), _t(img), torch.zeros(1, 2, H, W), torch.tensor(A[None], dtype=torch.float32),
                          torch.zeros(1, 3, H, W), torch.zeros(1, 1, H, W))[0].permute(1, 2, 0).numpy()
    a = warp(o, A)[:, ::-1]
    b = warp(o[:, ::-1].copy(), flip_affine(A, W))
    inner = (slice(5, -5), slice(5, -5))
    assert np.abs(a[inner] - b[inner]).max() < 1e-4


def test_model_outputs():
    net = CorrectionMapNet(checkpoint=False).eval()
    rgb, aux = torch.rand(2, 3, 64, 48), torch.rand(2, N_AUX, 64, 48)
    with torch.no_grad():
        y = net(rgb, aux)
    assert y["flow"].shape == (2, 2, 64, 48) and y["gain"].shape == (2, 3, 64, 48)
    assert y["alpha"].shape == y["fill"].shape == (2, 1, 64, 48)
    # 처음엔 거의 아무것도 고치지 않음
    assert y["flow"].abs().max() == 0 and y["gain"].abs().max() == 0
    assert abs(float(y["alpha"].mean()) - 1 / (1 + np.e ** 2)) < 1e-4
