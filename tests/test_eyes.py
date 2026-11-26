"""정밀 측정 P3: 눈 꼭짓점·밑라인 (합성 눈으로 위치를 아는 상태에서 검사)."""
import cv2
import numpy as np

from photostory.core.measure.eyes import lower_lid, refine_outer_corner

SKIN, LINE, SCLERA, IRIS = 200, 70, 225, 60


def _eye(tail=True, flip=False):
    """사진 오른쪽 눈 모양: 안쪽 꼭짓점 (100,100), 실제 바깥 꼭짓점 (300,90).
    홍채 중심 (200,90) 반지름 25, 아랫눈꺼풀이 홍채를 y=108에서 가림. 그 아래 반사판 빛 띠."""
    g = np.full((200, 420), SKIN, np.uint8)
    inner, outer = (100, 100), (300, 90)
    xs = np.linspace(0, 1, 60)
    up = [(inner[0] + (outer[0] - inner[0]) * t, inner[1] + (outer[1] - inner[1]) * t - 38 * np.sin(np.pi * t)) for t in xs]
    lo = [(inner[0] + (outer[0] - inner[0]) * t, inner[1] + (outer[1] - inner[1]) * t + 14 * np.sin(np.pi * t)) for t in xs]
    poly = np.array(up + lo[::-1], np.int32)
    cv2.fillPoly(g, [poly], SCLERA)
    iris = np.zeros_like(g)
    cv2.circle(iris, (200, 90), 25, 255, -1)
    inside = np.zeros_like(g)
    cv2.fillPoly(inside, [poly], 255)
    g[(iris > 0) & (inside > 0)] = IRIS
    cv2.polylines(g, [np.array(up, np.int32)], False, LINE, 3)
    cv2.polylines(g, [np.array(lo, np.int32)], False, LINE, 2)
    if tail:   # 위로 뻗는 아이라인 꼬리 (기준 아님)
        cv2.line(g, (300, 90), (345, 62), LINE, 3)
    g[118:126, 185:215] = 250   # 반사판 빛 띠 (밑라인으로 착각하면 안 됨)
    g = cv2.GaussianBlur(g, (0, 0), 1.0).astype(np.float32)
    if flip:
        g = np.ascontiguousarray(g[:, ::-1])
    return g


def test_corner_found_and_tail_ignored():
    g = _eye()
    f = refine_outer_corner(g, (288, 97), (100, 100))     # 모델 점: 실제보다 아래·안쪽
    assert f.confidence >= 0.5
    assert abs(f.point[0] - 300) <= 5 and abs(f.point[1] - 90) <= 4, f


def test_corner_mirrored_eye():
    g = _eye(flip=True)                                  # 사진 왼쪽 눈
    w = g.shape[1] - 1
    f = refine_outer_corner(g, (w - 288, 97), (w - 100, 100))
    assert f.confidence >= 0.5
    assert abs(f.point[0] - (w - 300)) <= 5 and abs(f.point[1] - 90) <= 4, f


def test_corner_unclear_gives_low_confidence():
    g = np.full((200, 420), SKIN, np.float32)            # 눈 틈이 안 보이는 사진
    f = refine_outer_corner(g, (288, 97), (100, 100))
    assert f.confidence < 0.5


def test_lower_lid_not_fooled_by_reflection():
    g = _eye()
    f = lower_lid(g, (200, 90), (200, 115), 25)
    lid_y = 100 - 10 / 2 + 14   # 아래 곡선 x=200 근처: 중간 높이 95 + 14 → 약 109
    assert abs(f.point[1] - lid_y) <= 3, f
    assert f.confidence >= 0.4
