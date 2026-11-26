"""손님 한 명의 작업 상태 (설계서 1-1, 2절, 6절).

다음 손님으로 넘어가면 Session 객체를 통째로 버린다. 파일로는 아무것도 저장하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Gender(str, Enum):
    MALE = "남"
    FEMALE = "여"


# 강도 부위 (설계서 2-3 표). id → 화면 이름
PARTS: dict[str, str] = {
    "skin": "피부",
    "tone": "명암",
    "eyes": "눈",
    "brows": "눈썹",
    "nose": "코",
    "mouth": "입",
    "jaw": "얼굴형·턱",
    "neck": "목·어깨",
    "hair": "머리카락",
}

# 기본 모드 3단계 (설계서 2-2). 샘플로 결과를 보고 조정한다.
LEVELS: dict[str, float] = {"약하게": 0.6, "기본": 1.0, "강하게": 1.4}
STRENGTH_MIN, STRENGTH_MAX = 0.0, 2.0


@dataclass
class Strength:
    """부위별 배율. 1.0 = 학습 모델(지금은 규칙 방식)이 정한 보정량."""
    values: dict[str, float] = field(default_factory=lambda: {k: 1.0 for k in PARTS})
    level: str | None = "기본"   # 기본 모드에서 고른 단계. 전문가 모드에서 따로 바꾸면 None

    @classmethod
    def from_level(cls, level: str) -> "Strength":
        v = LEVELS[level]
        return cls({k: v for k in PARTS}, level)

    def get(self, part: str) -> float:
        return float(self.values.get(part, 1.0))

    def set(self, part: str, value: float) -> None:
        self.values[part] = min(STRENGTH_MAX, max(STRENGTH_MIN, round(value, 2)))
        lv = {round(x, 2) for x in self.values.values()}
        self.level = next((n for n, x in LEVELS.items() if lv == {x}), None)

    def set_all(self, level: str) -> None:
        v = LEVELS[level]
        self.values = {k: v for k in PARTS}
        self.level = level

    def copy(self) -> "Strength":
        return Strength(dict(self.values), self.level)


@dataclass
class Customer:
    name: str = ""
    purposes: list[str] = field(default_factory=list)   # 규격 id
    gender: Gender = Gender.FEMALE
    glasses: bool = False
    strength: Strength = field(default_factory=Strength)

    def validate(self) -> list[str]:
        errs = []
        if not self.name.strip():
            errs.append("이름을 입력하세요")
        bad = set('\\/:*?"<>|')
        if any(c in bad for c in self.name):
            errs.append('이름에 \\ / : * ? " < > | 는 쓸 수 없습니다')
        if not self.purposes:
            errs.append("용도를 한 개 이상 고르세요")
        return errs


@dataclass
class Placement:
    """전송본 배치 미세 조정 (설계서 2-4). 규격마다 따로."""
    scale: float = 0.0    # -0.10 ~ +0.10 (얼굴 크기)
    shift: float = 0.0    # -0.05 ~ +0.05 (위아래, 사진 높이 대비)
    chin_bias: float = 0.0  # 0 = 턱 최소선, 1 = 턱 최대선 (머리 볼륨이 클수록 1 쪽)


@dataclass
class Session:
    customer: Customer = field(default_factory=Customer)
    original: Path | None = None
    glasses_photo: Path | None = None
    placements: dict[str, Placement] = field(default_factory=dict)
    disabled_steps: set[str] = field(default_factory=set)
    crease_ref: str = "auto"   # 쌍꺼풀 기준 눈: "auto"(더 또렷한 눈) / "왼눈" / "오른눈" (가이드 §5-3)

    def ready_to_start(self) -> bool:
        if self.original is None:
            return False
        return not self.customer.glasses or self.glasses_photo is not None

    def placement(self, spec_id: str) -> Placement:
        return self.placements.setdefault(spec_id, Placement())

    def swap_photos(self) -> None:
        self.original, self.glasses_photo = self.glasses_photo, self.original
