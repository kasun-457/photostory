"""변형 엔진 (core/warp.py)."""
import cv2
import numpy as np

from photostory.core.warp import Warp, gaussian_weight


def _dot_image(x=100, y=80):
    img = np.full((200, 240, 3), 200, np.uint8)
    cv2.circle(img, (x, y), 4, (20, 20, 20), -1, cv2.LINE_AA)
    return img


def _centroid(img):
    d = (img[..., 0] < 100).astype(np.float32)
    ys, xs = np.nonzero(d)
    return xs.mean(), ys.mean()


def _piece(shape, offset, center, r=40, feather=10):
    m = np.zeros(shape, np.float32)
    cv2.circle(m, (int(center[0] - offset[0]), int(center[1] - offset[1])), r, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), feather)


def test_shift_moves_pixels_and_points():
    img = _dot_image()
    w = Warp(img.shape, (20, 10, 220, 190))
    wt = _piece(w.roi_shape, w.offset(), (100, 80))
    w.shift(wt, 6, -3)
    out = w.apply_image(img)
    cx, cy = _centroid(out)
    assert abs(cx - 106) < 0.6 and abs(cy - 77) < 0.6
    (px, py), = w.forward_points(np.array([[100, 80]]))
    assert abs(px - 106) < 0.3 and abs(py - 77) < 0.3
    assert (out[5, 5] == img[5, 5]).all()             # 범위 밖은 그대로


def test_two_shifts_compose():
    img = _dot_image()
    w = Warp(img.shape, (0, 0, 240, 200))
    for _ in range(2):
        p = w.forward_points(np.array([[100, 80]]))[0]
        w.shift(_piece(w.roi_shape, w.offset(), p), 3, 0)
    cx, _ = _centroid(w.apply_image(img))
    assert abs(cx - 106) < 0.6


def test_rotate_piece():
    img = np.full((200, 240, 3), 200, np.uint8)
    cv2.line(img, (80, 100), (160, 100), (0, 0, 0), 3)
    w = Warp(img.shape, (0, 0, 240, 200))
    m = np.zeros(w.roi_shape, np.float32)
    m[60:140, 60:180] = 1
    w.rotate(cv2.GaussianBlur(m, (0, 0), 6), (120, 100), 5.0)   # 시계 방향 5°
    out = w.apply_image(img)
    ys, xs = np.nonzero(out[..., 0] < 100)
    sel = (xs > 95) & (xs < 145)
    slope = np.polyfit(xs[sel], ys[sel], 1)[0]
    assert abs(np.degrees(np.arctan(slope)) - 5.0) < 0.6       # 오른쪽이 내려감


def test_gaussian_weight_peak():
    g = gaussian_weight((50, 60), (10, 20), (40, 45), 5)
    assert abs(g.max() - 1.0) < 1e-6 and g[25, 30] == g.max()
