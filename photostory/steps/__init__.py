"""보정 단계 목록. 순서는 설계서 3-2 표(가이드 순서)를 따른다.

새 단계는 파일 하나를 추가하고 아래 목록의 알맞은 자리에 넣는다.
"""
from __future__ import annotations

from ..core.pipeline import Step
from .a4 import A4Step
from .catchlight import CatchlightStep
from .background import BackgroundStep
from .guides import GuidesStep
from .level import LevelStep
from .sharpen import SharpenStep
from .skin import SkinStep
from .symmetry import SymmetryStep
from .tone import ToneStep
from .transfer import TransferStep


def default_steps() -> list[Step]:
    return [
        LevelStep(),        # 1
        GuidesStep(),       # 2
        SkinStep(),        # 3·4 (규칙 방식: 액션 '수정-01')
        CatchlightStep(),  # 5 (반사로 뚫린 눈이 한쪽에만 있을 때)
        SharpenStep(),     # 6 (규칙 방식: 액션 '수정-02', 흐릴 때만)
        ToneStep(),        # 7 (규칙 방식: 액션 '닷징버닝')
        SymmetryStep(),    # 8 (§9 유동화 + 부분 이동)
        # 9~12: 쌍꺼풀·목·머리카락·안경 (2단계)
        BackgroundStep(),   # 13
        # 14~15: 머리카락 밝기, 어깨 밸런스 (1·2단계)
        TransferStep(),     # 16 (수정본은 이 단계 직전 이미지)
        A4Step(),           # 17
    ]
