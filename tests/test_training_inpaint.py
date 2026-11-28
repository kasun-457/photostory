"""3-5 부분 생성형 학습 도구 테스트 (합성 데이터)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.inpaint_data import synth_mask, telea_baseline  # noqa: E402
from training.inpaint_model import InpaintNet, PatchDisc, compose  # noqa: E402
from training.train_inpaint import gen_losses  # noqa: E402


def test_synth_mask_size_and_nonempty():
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = synth_mask(rng, 128)
        assert m.shape == (128, 128) and m.dtype == np.uint8
        assert 0 < m.mean() < 0.6


def test_model_shapes_and_compose_keeps_outside():
    net, d = InpaintNet(widths=(16, 32, 64, 128), blocks=2), PatchDisc(16)
    img = torch.rand(2, 3, 96, 96)
    m = torch.zeros(2, 1, 96, 96)
    m[:, :, 30:60, 30:60] = 1
    pred = net(img, m)
    assert pred.shape == img.shape and float(pred.min()) >= 0 and float(pred.max()) <= 1
    c = compose(img, m, pred)
    assert torch.equal(c[:, :, :30], img[:, :, :30])
    assert d(c, m).shape[0] == 2


def test_model_does_not_see_hole_content():
    net = InpaintNet(widths=(16, 32, 64, 128), blocks=2).eval()
    img = torch.rand(1, 3, 64, 64)
    m = torch.zeros(1, 1, 64, 64)
    m[:, :, 20:40, 20:40] = 1
    img2 = img.clone()
    img2[:, :, 20:40, 20:40] = 0.123          # 구멍 속만 다름
    with torch.no_grad():
        assert torch.allclose(net(img, m), net(img2, m), atol=1e-6)


def test_losses_and_one_step_learns():
    torch.manual_seed(0)
    net = InpaintNet(widths=(16, 32, 64, 128), blocks=1)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    img = torch.full((2, 3, 64, 64), 0.6)
    m = torch.zeros(2, 1, 64, 64)
    m[:, :, 16:48, 16:48] = 1
    b = dict(img=img, mask=m, loss=torch.ones_like(m))
    first = None
    for _ in range(30):
        ls = gen_losses(net(img, m), b)
        first = first or float(ls["hole"])
        opt.zero_grad()
        ls["total"].backward()
        opt.step()
    assert float(ls["hole"]) < first * 0.5


def test_telea_baseline_fills_flat_hole():
    img = np.full((40, 40, 3), 0.5, np.float32)
    m = np.zeros((40, 40), np.float32)
    m[10:20, 10:20] = 1
    out = telea_baseline(img * (1 - m[..., None]), m)
    assert abs(float(out[15, 15].mean()) - 0.5) < 0.02
