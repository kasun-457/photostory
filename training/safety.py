"""원본 사진 폴더 보호 (학습설계 0-1).

학습 도구는 원본 폴더(그리고 D: 드라이브 전체)를 **읽기만** 한다.
쓰기 전에 반드시 `check_output()`으로 출력 위치를 검사하고, 원본 파일은 `open_source()`로만 연다.
"""
from __future__ import annotations

import os
from pathlib import Path

# 컴퓨터마다 경로가 다르면 환경 변수로 바꾼다 (학습설계 4-1):
#   PHOTOSTORY_SRC  = 원본 사진 폴더 (년/월/찍은 사람 폴더가 들어 있는 곳)
#   PHOTOSTORY_WORK = 작업 폴더 (조각·정답 지도·학습 결과. git에 올리지 않음)
DEFAULT_SOURCE = Path(os.environ.get("PHOTOSTORY_SRC") or r"D:\1_촬영\3_증명사진")
DEFAULT_WORK = Path(os.environ.get("PHOTOSTORY_WORK") or r"C:\photostory_data")
# 사용자 요청 (2026-11-27): D 드라이브 안의 파일은 손상·변형되면 안 된다 → 드라이브 통째로 막는다.
# 원본 폴더가 다른 드라이브에 있으면 그 드라이브도 통째로 막는다 (작업 폴더와 같은 드라이브면 원본 폴더만).
_src_drive = DEFAULT_SOURCE.anchor
_other_drive = bool(_src_drive) and os.path.normcase(_src_drive) != os.path.normcase(DEFAULT_WORK.anchor)
PROTECTED_ROOTS = tuple(dict.fromkeys(
    [Path("D:/")] + ([Path(_src_drive)] if _other_drive else []) + [DEFAULT_SOURCE]))


class UnsafeOutput(Exception):
    pass


def _norm(p: Path | str) -> Path:
    return Path(os.path.abspath(os.path.normcase(str(p))))


def is_inside(child: Path | str, parent: Path | str) -> bool:
    c, p = _norm(child), _norm(parent)
    return c == p or p in c.parents


def check_output(out: Path | str, sources: tuple[Path | str, ...] = ()) -> Path:
    """출력 위치가 원본 폴더나 보호 드라이브 아래면 멈춘다. 통과하면 절대 경로를 돌려준다."""
    out = Path(os.path.abspath(out))
    for root in (*PROTECTED_ROOTS, *sources):
        if is_inside(out, root):
            raise UnsafeOutput(f"출력 위치가 보호된 원본 폴더 안입니다: {out} (보호: {root})")
    return out


def open_source(path: Path | str):
    """원본 파일은 읽기 전용('rb')으로만 연다."""
    return open(path, "rb")
