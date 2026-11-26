import numpy as np
import pytest

from photostory.models.hub import ModelHub


@pytest.fixture(scope="session")
def hub():
    h = ModelHub()
    if h.missing_files():
        pytest.skip("모델 파일 없음: python tools/fetch_models.py")
    yield h
    h.close()


@pytest.fixture(scope="session")
def portrait(hub):
    """테스트용 증명사진 흉내: 공개 사진(scikit-image astronaut, NASA)의 사람만 오려
    사진관 배경처럼 약간 어두운 흰색(244,242,238) 위에 얹고, 위아래 여백을 붙인 세로 사진."""
    skd = pytest.importorskip("skimage.data")
    import cv2
    img = cv2.resize(skd.astronaut(), (1536, 1536), interpolation=cv2.INTER_CUBIC)
    m = hub.get("person_segmentation").segment(img).person
    m = cv2.GaussianBlur((m > 0.5).astype(np.float32), (0, 0), 2)[..., None]
    bg = np.empty_like(img)
    bg[:] = (244, 242, 238)
    comp = (img * m + bg * (1 - m)).astype(np.uint8)
    canvas = np.empty((2304, 1536, 3), np.uint8)
    canvas[:] = (244, 242, 238)
    canvas[400:400 + 1536] = comp
    return np.ascontiguousarray(canvas)
