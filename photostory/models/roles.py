"""역할별 표준 틀 (설계서 6-2).

보정 단계는 모델 이름을 모른다. 역할(얼굴점, 사람 영역 …)만 요청하고,
점은 모델 번호가 아니라 아래 표준 이름으로 받는다.

좌우 표기: **사진 기준**이다. `왼` = 사진의 왼쪽(손님의 오른쪽).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

# 얼굴 점 표준 이름. 어댑터는 최소한 REQUIRED_POINTS를 채워야 한다.
REQUIRED_POINTS = (
    "왼눈_바깥꼭짓점", "오른눈_바깥꼭짓점",
    "왼눈_안쪽꼭짓점", "오른눈_안쪽꼭짓점",
    "턱끝", "이마_위",
)
OPTIONAL_POINTS = (
    "왼눈_홍채중심", "오른눈_홍채중심",
    "왼눈_윗꺼풀", "오른눈_윗꺼풀", "왼눈_아랫꺼풀", "오른눈_아랫꺼풀",
    "왼눈썹_바깥", "오른눈썹_바깥", "왼눈썹_안쪽", "오른눈썹_안쪽", "왼눈썹_위", "오른눈썹_위",
    "코끝", "코밑", "왼콧볼", "오른콧볼",
    "왼입꼬리", "오른입꼬리", "윗입술_중심", "아랫입술_중심",
    "왼볼_입높이", "오른볼_입높이", "왼턱선", "오른턱선", "왼턱끝쪽", "오른턱끝쪽",
    "왼얼굴_옆", "오른얼굴_옆",
)
POSE_POINTS = ("왼어깨", "오른어깨", "코_자세")


@dataclass
class FaceResult:
    points: dict[str, tuple[float, float]]          # 입력 이미지 px 좌표
    confidence: float = 1.0
    all_points: np.ndarray | None = None             # 모델 고유 전체 점 (N, 2), 디버그·확장용
    extra: dict = field(default_factory=dict)


@dataclass
class SegmentResult:
    person: np.ndarray        # float32 0~1, 입력 이미지 크기
    hair: np.ndarray | None = None
    face_skin: np.ndarray | None = None
    body_skin: np.ndarray | None = None
    clothes: np.ndarray | None = None
    accessories: np.ndarray | None = None


@dataclass
class PoseResult:
    points: dict[str, tuple[float, float]]
    confidence: float = 1.0


class Role(ABC):
    """모든 역할 어댑터의 공통 부모. 무거운 초기화는 load()에서 한다."""
    role: str = ""

    def __init__(self, config: dict):
        self.config = config

    def load(self) -> None:  # 모델 미리 읽기
        pass

    def close(self) -> None:
        pass


class FaceLandmarks(Role):
    role = "face_landmarks"

    @abstractmethod
    def detect(self, rgb: np.ndarray) -> FaceResult | None: ...


class PersonSegmentation(Role):
    role = "person_segmentation"

    @abstractmethod
    def segment(self, rgb: np.ndarray) -> SegmentResult: ...


class PoseLandmarks(Role):
    role = "pose_landmarks"

    @abstractmethod
    def detect(self, rgb: np.ndarray) -> PoseResult | None: ...


class CorrectionMap(Role):
    """학습 모델(보정 지도). 처음엔 가이드 규칙 어댑터, 학습 후 학습 모델 어댑터로 교체 (1·3단계)."""
    role = "correction_map"

    @abstractmethod
    def predict(self, rgb: np.ndarray, points: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]: ...


ROLE_TYPES: dict[str, type[Role]] = {
    c.role: c for c in (FaceLandmarks, PersonSegmentation, PoseLandmarks, CorrectionMap)
}
