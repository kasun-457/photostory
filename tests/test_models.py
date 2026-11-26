from photostory.core.prepare import analyze, quick_face_check
from photostory.models.roles import REQUIRED_POINTS


def test_face_and_mask(hub, portrait):
    assert quick_face_check(portrait, hub).ok
    a = analyze("", hub, rgb=portrait)
    for k in REQUIRED_POINTS:
        assert k in a.points
    lx, rx = a.points["왼눈_바깥꼭짓점"][0], a.points["오른눈_바깥꼭짓점"][0]
    assert lx < rx  # 사진 기준 좌우
    assert a.person_mask.shape == portrait.shape[:2]
    assert a.points["이마_위"][1] < a.points["턱끝"][1]
