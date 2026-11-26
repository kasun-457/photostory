"""규격 데이터 (설계서 4-1, 가이드 §15-1·§16).

규격 하나 = JSON 파일 하나. 기본 14종은 패키지 안 `specs/`에 있고,
사진관이 추가·수정한 규격은 사용자 설정 폴더의 `specs/`에 저장한다.
같은 id가 있으면 사용자 쪽이 우선이다. 기본 규격은 지우지 않고 숨기기만 한다.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

BUILTIN_DIR = Path(__file__).resolve().parent.parent / "specs"
A4_PX = (2480, 3508)
A4_FIRST_X = 146
A4_GAP = 30


def cm_to_px(cm: float, dpi: float = 300) -> float:
    return cm / 2.54 * dpi


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


@dataclass
class Spec:
    id: str
    name: str
    size_cm: list[float]
    head_cm: float
    top_cm: float = 0.2
    dpi: float = 300
    chin_margin_cm: float = 0.1
    background: str = "#ffffff"
    a4_count: int | None = None
    a4_label: str = ""
    file_label: str = ""
    aliases: list[str] = field(default_factory=list)
    order: int = 100
    override_px: dict | None = None   # 300dpi 기준 실제 템플릿 값 (여권)
    note: str = ""
    builtin: bool = False

    # ---- 300dpi 기준이 아닌 경우를 포함한 픽셀 값 ----
    @property
    def size_px(self) -> tuple[int, int]:
        return (round(cm_to_px(self.size_cm[0], self.dpi)), round(cm_to_px(self.size_cm[1], self.dpi)))

    @property
    def size_px_300(self) -> tuple[int, int]:
        return (round(cm_to_px(self.size_cm[0])), round(cm_to_px(self.size_cm[1])))

    def _line(self, key: str, value_cm: float) -> float:
        k = self.dpi / 300
        if self.override_px and key in self.override_px:
            return self.override_px[key] * k
        return round(cm_to_px(value_cm), 1) * k

    @property
    def crown_line(self) -> float:
        return self._line("crown", self.top_cm)

    @property
    def chin_min_line(self) -> float:
        return self._line("chin_min", self.top_cm + self.head_cm)

    @property
    def chin_max_line(self) -> float:
        if self.override_px and "chin_max" in self.override_px:
            return self._line("chin_max", 0)
        return self.chin_min_line + round(cm_to_px(self.chin_margin_cm), 1) * self.dpi / 300

    @property
    def center_x(self) -> float:
        return self.size_px[0] / 2

    @property
    def bg_rgb(self) -> tuple[int, int, int]:
        return hex_to_rgb(self.background)

    @property
    def is_white(self) -> bool:
        return self.background.lower() in ("#ffffff", "#fff", "white")

    def a4_count_resolved(self) -> int:
        if self.a4_count:
            return int(self.a4_count)
        w = self.size_px_300[0]
        return max(1, math.floor((A4_PX[0] - A4_FIRST_X + A4_GAP) / (w + A4_GAP)))

    @property
    def label(self) -> str:
        return self.a4_label or self.name

    @property
    def file_tag(self) -> str:
        return self.file_label or re.sub(r"[\\/:*?\"<>|\s]", "", self.name)

    def validate(self) -> list[str]:
        errs = []
        if not self.id or not re.fullmatch(r"[A-Za-z0-9_\-]+", self.id):
            errs.append("id는 영문·숫자·_·- 만 쓸 수 있습니다")
        if not self.name.strip():
            errs.append("이름을 입력하세요")
        if len(self.size_cm) != 2 or min(self.size_cm) <= 0:
            errs.append("크기(cm)를 확인하세요")
        if self.head_cm <= 0:
            errs.append("정수리~턱 길이를 확인하세요")
        if self.top_cm < 0:
            errs.append("머리 윗공간은 0 이상이어야 합니다")
        if len(self.size_cm) == 2 and self.top_cm + self.head_cm >= self.size_cm[1]:
            errs.append("머리 윗공간 + 정수리~턱이 사진 높이보다 큽니다")
        if self.dpi < 72 or self.dpi > 1200:
            errs.append("dpi는 72~1200 사이여야 합니다")
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", self.background):
            errs.append("배경색은 #rrggbb 형식이어야 합니다")
        if self.a4_count is not None:
            w = self.size_px_300[0]
            if A4_FIRST_X + self.a4_count * w + (self.a4_count - 1) * A4_GAP > A4_PX[0]:
                errs.append("A4 장수가 한 줄에 들어가지 않습니다")
        return errs

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("builtin", None)
        return d

    @classmethod
    def from_json(cls, d: dict, builtin: bool = False) -> "Spec":
        known = {f for f in cls.__dataclass_fields__}
        s = cls(**{k: v for k, v in d.items() if k in known})
        s.builtin = builtin
        return s


class SpecRegistry:
    def __init__(self, user_dir: Path | None = None, hidden: set[str] | None = None):
        self.user_dir = user_dir
        self.hidden: set[str] = set(hidden or ())
        self._builtin: dict[str, Spec] = {}
        self._user: dict[str, Spec] = {}
        self.reload()

    def reload(self) -> None:
        self._builtin = {s.id: s for s in self._load_dir(BUILTIN_DIR, True)}
        self._user = {s.id: s for s in self._load_dir(self.user_dir, False)} if self.user_dir else {}

    @staticmethod
    def _load_dir(d: Path | None, builtin: bool) -> list[Spec]:
        out = []
        if d and d.is_dir():
            for f in sorted(d.glob("*.json")):
                try:
                    out.append(Spec.from_json(json.loads(f.read_text("utf-8")), builtin))
                except (ValueError, TypeError):
                    continue  # 깨진 파일은 건너뛴다
        return out

    def all(self, include_hidden: bool = False) -> list[Spec]:
        merged = dict(self._builtin)
        merged.update(self._user)
        specs = [s for s in merged.values() if include_hidden or s.id not in self.hidden]
        return sorted(specs, key=lambda s: (s.order, s.name))

    def get(self, sid: str) -> Spec:
        if sid in self._user:
            return self._user[sid]
        return self._builtin[sid]

    def find(self, text: str) -> Spec | None:
        """용도 이름이 조금 달라도 같은 규격을 찾는다 (가이드 §15-1)."""
        t = text.replace(" ", "")
        for s in self.all(include_hidden=True):
            names = [s.id, s.name, s.a4_label, s.file_label, *s.aliases]
            if any(t == n.replace(" ", "") for n in names if n):
                return s
        return None

    def save(self, spec: Spec) -> None:
        errs = spec.validate()
        if errs:
            raise ValueError("\n".join(errs))
        if not self.user_dir:
            raise RuntimeError("사용자 규격 폴더가 설정되지 않았습니다")
        self.user_dir.mkdir(parents=True, exist_ok=True)
        (self.user_dir / f"{spec.id}.json").write_text(
            json.dumps(spec.to_json(), ensure_ascii=False, indent=1), "utf-8")
        self.reload()

    def delete(self, sid: str) -> None:
        """사용자 규격은 지우고, 기본 규격은 숨긴다."""
        if sid in self._user and self.user_dir:
            (self.user_dir / f"{sid}.json").unlink(missing_ok=True)
        if sid in self._builtin:
            self.hidden.add(sid)
        self.reload()

    def reset_builtin(self, sid: str) -> None:
        """기본값으로 되돌리기: 사용자 수정본을 지우고 숨김을 푼다."""
        if sid in self._user and sid in self._builtin and self.user_dir:
            (self.user_dir / f"{sid}.json").unlink(missing_ok=True)
        self.hidden.discard(sid)
        self.reload()
