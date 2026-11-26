"""출력 (설계서 4절): 파일 이름, 같은 이름 처리, 저장."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from .io.imageio import write_jpeg
from .specs import Spec


class Conflict(str, Enum):
    OVERWRITE = "덮어쓰기"
    NUMBER = "번호 붙이기"
    CANCEL = "취소"


@dataclass
class OutputFile:
    path: Path
    image: np.ndarray
    dpi: float


def planned_names(name: str, specs: list[Spec]) -> list[tuple[str, str]]:
    """(종류, 파일 이름) 목록. 종류: retouched / transfer:{id} / a4"""
    out = [("retouched", f"{name}_수정본.jpg")]
    if len(specs) == 1:
        out.append((f"transfer:{specs[0].id}", f"{name}_전송.jpg"))
    else:
        out += [(f"transfer:{s.id}", f"{name}_전송_{s.file_tag}.jpg") for s in specs]
    out.append(("a4", f"{name}_a4출력.jpg"))
    return out


def existing(folder: Path, names: list[str]) -> list[str]:
    return [n for n in names if (folder / n).exists()]


def numbered(folder: Path, names: list[str]) -> list[str]:
    """세 종류 파일에 같은 번호를 붙인다: 홍길동_수정본(2).jpg …"""
    n = 2
    while True:
        cand = [f"{Path(x).stem}({n}){Path(x).suffix}" for x in names]
        if not existing(folder, cand):
            return cand
        n += 1


def resolve(folder: Path, names: list[str], choice: Conflict) -> list[str] | None:
    if not existing(folder, names) or choice == Conflict.OVERWRITE:
        return names
    if choice == Conflict.NUMBER:
        return numbered(folder, names)
    return None


def save_all(folder: Path, name: str, specs: list[Spec], outputs: dict,
             choice: Conflict = Conflict.CANCEL) -> list[Path] | None:
    plan = planned_names(name, specs)
    names = resolve(folder, [n for _, n in plan], choice)
    if names is None:
        return None
    folder.mkdir(parents=True, exist_ok=True)
    by_id = {s.id: s for s in specs}
    saved = []
    for (kind, _), fname in zip(plan, names):
        if kind == "retouched":
            img, dpi = outputs["retouched"], 300
        elif kind == "a4":
            img, dpi = outputs["a4"], 300
        else:
            sid = kind.split(":", 1)[1]
            img, dpi = outputs["transfers"][sid], by_id[sid].dpi
        saved.append(write_jpeg(folder / fname, img, dpi=dpi))
    return saved
