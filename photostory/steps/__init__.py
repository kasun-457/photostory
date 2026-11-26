"""보정 단계 목록. 순서는 설계서 3-2 표(가이드 순서)를 따른다.

새 단계는 파일 하나를 추가하고 아래 목록의 알맞은 자리에 넣는다.
"""
from __future__ import annotations

from ..core.pipeline import Step
from .a4 import A4Step
from .catchlight import CatchlightStep
from .background import BackgroundStep
from .guides import GuidesStep
from .hair import HairBrightnessStep, HairTidyStep
from .level import LevelStep
from .neck import NeckStep
from .sharpen import SharpenStep
from .shoulder import ShoulderStep
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
        # 9: 쌍꺼풀 (2단계)
        NeckStep(),        # 10 (§10 목 20px 미만 유동화 / 이상 몸 회전)
        HairTidyStep(),    # 11 (§11-3 잔머리. 핀·머리끈은 아직)
        # 12: 안경 (2단계)
        BackgroundStep(),   # 13
        HairBrightnessStep(),  # 14 (§13 너무 밝을 때만)
        ShoulderStep(),    # 15 (§14 규격별 트리밍 경계에서 어깨 높이)
        TransferStep(),     # 16 (수정본은 이 단계 직전 이미지)
        A4Step(),           # 17
    ]
