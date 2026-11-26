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
def portrait():
    """테스트용 공개 사진 (scikit-image astronaut, NASA 공개 사진)을 흰 배경 증명사진 비슷하게 키운 것."""
    skd = pytest.importorskip("skimage.data")
    import cv2
    img = skd.astronaut()                       # 512x512
    img = cv2.resize(img, (1536, 1536), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(img)
