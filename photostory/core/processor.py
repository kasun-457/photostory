"""손님 한 명 처리 흐름을 화면과 분리해 묶는다 (화면은 이 객체만 호출한다).

session → RenderParams 변환, prepare 캐시(경로별), 미리보기/원본 render.
"""
from __future__ import annotations

import threading
from pathlib import Path

from ..models.hub import ModelHub
from ..steps import default_steps
from ..steps.background import background_rgb_for
from .pipeline import Analysis, CancelToken, Pipeline, RenderParams, RenderResult
from .prepare import analyze
from .session import PARTS, Session
from .specs import Spec, SpecRegistry


class Processor:
    def __init__(self, hub: ModelHub, specs: SpecRegistry, font: Path | None = None):
        self.hub = hub
        self.specs = specs
        self.font = font
        self.pipeline = Pipeline(default_steps())
        self._analyses: dict[Path, Analysis] = {}
        self._events: dict[Path, threading.Event] = {}
        self._errors: dict[Path, Exception] = {}
        self._lock = threading.Lock()

    # ---------- 손님 바뀜
    def clear(self) -> None:
        with self._lock:
            self._analyses.clear()
            self._events.clear()
            self._errors.clear()
        self.pipeline.reset()

    # ---------- prepare (업로드하자마자 뒤에서 시작)
    def prepare(self, path: Path, token: CancelToken | None = None) -> Analysis:
        path = Path(path)
        with self._lock:
            if path in self._analyses:
                return self._analyses[path]
            ev = self._events.get(path)
            owner = ev is None
            if owner:
                ev = self._events[path] = threading.Event()
        if not owner:
            ev.wait()
            with self._lock:
                if path in self._errors:
                    raise self._errors[path]
                return self._analyses[path]
        try:
            a = analyze(path, self.hub, token)
            with self._lock:
                self._analyses[path] = a
            return a
        except Exception as e:
            with self._lock:
                self._errors[path] = e
                self._events.pop(path, None)
            raise
        finally:
            ev.set()

    def start(self, session: Session, token: CancelToken | None = None) -> None:
        assert session.original is not None
        a = self.prepare(session.original, token)
        # 안경 사진(P5)은 2단계에서 원본 좌표에 맞춰 쓴다. 지금은 분석만 해 둔다.
        if session.customer.glasses and session.glasses_photo:
            try:
                self.prepare(session.glasses_photo, token)
            except Exception:
                pass
        self.pipeline.reset(a)

    # ---------- render
    def chosen_specs(self, session: Session) -> list[Spec]:
        out = []
        for sid in session.customer.purposes:
            try:
                out.append(self.specs.get(sid))
            except KeyError:
                continue
        return out

    def params(self, session: Session) -> RenderParams:
        specs = self.chosen_specs(session)
        c = session.customer
        return RenderParams(
            values={
                "name": c.name.strip(),
                "gender": c.gender.value,
                "glasses": c.glasses,
                "strength": dict(c.strength.values),
                # 단계는 자기 부위 키만 param_keys에 넣는다 → 다른 부위를 바꾸면 다시 계산하지 않음
                **{f"strength.{k}": round(c.strength.get(k), 2) for k in PARTS},
                "specs": specs,
                "placements": {s.id: session.placement(s.id) for s in specs},
                "bg_color": background_rgb_for(specs) if specs else (255, 255, 255),
                "font": str(self.font) if self.font else None,
            },
            disabled=frozenset(session.disabled_steps),
        )

    def render(self, session: Session, preview: bool, token: CancelToken | None = None,
               generation: int | None = None, progress=None) -> RenderResult:
        return self.pipeline.render(self.params(session), preview, token, generation, progress)

    def original_preview(self):
        """결과 화면 비교용 원본 (미리보기 해상도)."""
        return self.pipeline._base_frame(True).image
