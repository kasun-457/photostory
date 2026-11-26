"""설정 (설계서 5절). 사진관 PC에 JSON 한 파일로 저장한다. 손님 사진·보정 과정은 저장하지 않는다."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core.session import LEVELS, PARTS


def config_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "PhotoStory"


@dataclass
class AppSettings:
    expert: bool = False
    default_level: str = "기본"
    default_parts: dict[str, float] = field(default_factory=lambda: {k: 1.0 for k in PARTS})
    output_dir: str = ""
    fixed_output: bool = False
    font_path: str = ""                # 비우면 기본 글꼴(나눔고딕 Bold)
    hidden_specs: list[str] = field(default_factory=list)
    disabled_steps: list[str] = field(default_factory=list)
    last_purposes: list[str] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return config_dir() / "settings.json"

    @property
    def specs_dir(self) -> Path:
        return config_dir() / "specs"

    @classmethod
    def load(cls) -> "AppSettings":
        s = cls()
        try:
            d = json.loads(s.path.read_text("utf-8"))
            known = cls.__dataclass_fields__
            s = cls(**{k: v for k, v in d.items() if k in known})
        except (OSError, ValueError, TypeError):
            pass
        if s.default_level not in LEVELS:
            s.default_level = "기본"
        return s

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), "utf-8")
        os.replace(tmp, self.path)
