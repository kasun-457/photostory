"""보정 단계 목록. 순서는 설계서 3-2 표(가이드 순서)를 따른다.

새 단계는 파일 하나를 추가하고 아래 목록의 알맞은 자리에 넣는다.
"""
from __future__ import annotations

from ..core.pipeline import Step
from .a4 import A4Step
from .background import BackgroundStep
from .guides import GuidesStep
from .level import LevelStep
from .transfer import TransferStep


def default_steps() -> list[Step]:
    return [
        LevelStep(),        # 1
        GuidesStep(),       # 2
        # 3~12: 피부·명암·대칭·목·머리카락·안경 (1·2단계)
        BackgroundStep(),   # 13
        # 14~15: 머리카락 밝기, 어깨 밸런스 (1·2단계)
        TransferStep(),     # 16 (수정본은 이 단계 직전 이미지)
        A4Step(),           # 17
    ]
