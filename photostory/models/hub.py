"""모델 허브: 역할 이름으로 어댑터를 꺼내 준다 (설계서 6-2).

- models.yaml에서 역할 → 어댑터 클래스·파일을 읽는다.
- 어댑터는 처음 요청할 때 만들고 끝까지 쓴다. preload()로 프로그램 시작 때 뒤에서 미리 읽는다.
- 모델 파일은 `models/files/`(개발) 또는 설정에서 지정한 폴더에 둔다.
"""
from __future__ import annotations

import importlib
import logging
import threading
from pathlib import Path
from typing import TypeVar

import yaml

from .roles import ROLE_TYPES, Role

log = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "models.yaml"
DEFAULT_FILES = HERE / "files"

R = TypeVar("R", bound=Role)


class ModelMissing(Exception):
    pass


class ModelHub:
    def __init__(self, config_path: Path = DEFAULT_CONFIG, files_dir: Path = DEFAULT_FILES):
        self.config_path = config_path
        self.files_dir = files_dir
        self.roles: dict[str, dict] = yaml.safe_load(config_path.read_text("utf-8"))["roles"]
        self._instances: dict[str, Role] = {}
        self._lock = threading.Lock()

    def file_path(self, role: str) -> Path:
        return self.files_dir / self.roles[role]["file"]

    def missing_files(self) -> list[str]:
        return [r for r, c in self.roles.items() if c.get("file") and not self.file_path(r).exists()]

    def get(self, role: str, expected: type[R] | None = None) -> R:
        with self._lock:
            inst = self._instances.get(role)
            if inst is None:
                cfg = dict(self.roles[role])
                if cfg.get("file"):
                    cfg["path"] = self.file_path(role)
                    if not cfg["path"].exists():
                        raise ModelMissing(f"모델 파일이 없습니다: {cfg['file']} (tools/fetch_models.py 실행)")
                mod_name, cls_name = cfg["adapter"].rsplit(".", 1)
                mod = importlib.import_module(f"{__package__}.adapters.{mod_name}")
                cls = getattr(mod, cls_name)
                base = ROLE_TYPES.get(role)
                if base and not issubclass(cls, base):
                    raise TypeError(f"{cls_name}는 {role} 역할이 아닙니다")
                inst = cls(cfg)
                self._instances[role] = inst
        if expected is not None and not isinstance(inst, expected):
            raise TypeError(f"{role}: {type(inst).__name__}")
        return inst  # type: ignore[return-value]

    def preload(self, roles: list[str] | None = None) -> threading.Thread:
        def work():
            for r in roles or list(self.roles):
                try:
                    self.get(r).load()
                except Exception as e:  # 시작은 막지 않는다. 쓸 때 다시 오류를 낸다
                    log.warning("모델 미리 읽기 실패 %s: %s", r, e)
        t = threading.Thread(target=work, name="model-preload", daemon=True)
        t.start()
        return t

    def close(self) -> None:
        with self._lock:
            for inst in self._instances.values():
                inst.close()
            self._instances.clear()

    def non_commercial(self) -> list[str]:
        return [r for r, c in self.roles.items() if c.get("commercial") is not True]
