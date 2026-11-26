"""파이프라인 엔진 (설계서 3절, 6-1).

- prepare: 사진마다 1회 하는 무거운 분석 (읽기·인식·측정). 결과는 Analysis에 모은다.
- render : 보정 단계들을 순서대로 적용한다. 단계 결과는 캐시해서, 강도·배치를 바꾸면
           영향을 받는 단계부터만 다시 계산한다.
- 미리보기(긴 변 1,400px)와 원본 해상도를 같은 코드로 계산한다 (scale만 다름).
- 취소: CancelToken을 단계 사이마다 확인한다. 늦게 끝난 결과는 generation으로 걸러낸다.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)

PREVIEW_MAX_SIDE = 1400


class Cancelled(Exception):
    pass


class CancelToken:
    def __init__(self) -> None:
        self._ev = threading.Event()

    def cancel(self) -> None:
        self._ev.set()

    @property
    def cancelled(self) -> bool:
        return self._ev.is_set()

    def check(self) -> None:
        if self._ev.is_set():
            raise Cancelled()


# ---------------------------------------------------------------- 분석 결과

@dataclass
class Analysis:
    """prepare 결과. 좌표는 모두 원본 해상도 기준 (x, y)."""
    image: np.ndarray                           # 원본 RGB
    points: dict[str, tuple[float, float]]      # 표준 이름 → 좌표 (models/roles.py)
    person_mask: np.ndarray | None = None       # float32 0~1, 원본 크기
    hair_mask: np.ndarray | None = None
    confidence: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h


@dataclass
class Frame:
    """render 도중의 작업 상태. 단계는 Frame을 받아 새 Frame(또는 수정본)을 돌려준다."""
    image: np.ndarray
    points: dict[str, tuple[float, float]]
    person_mask: np.ndarray | None
    hair_mask: np.ndarray | None
    scale: float                                 # 원본 대비 배율 (미리보기 < 1)
    outputs: dict[str, Any] = field(default_factory=dict)  # 전송본, A4, 측정값 등
    notes: list[str] = field(default_factory=list)          # 건너뛴 단계, 남겨 둔 점

    def copy(self) -> "Frame":
        return Frame(self.image, dict(self.points), self.person_mask, self.hair_mask,
                     self.scale, dict(self.outputs), list(self.notes))

    def px(self, v: float) -> float:
        """원본 해상도 기준 px 값을 현재 해상도로."""
        return v * self.scale

    def warp(self, M: np.ndarray, size: tuple[int, int] | None = None,
             border_value: tuple[int, int, int] | None = None) -> "Frame":
        """이미지·마스크·점에 같은 2x3 아핀 변환을 적용한다."""
        h, w = self.image.shape[:2]
        size = size or (w, h)
        if border_value is None:
            img = cv2.warpAffine(self.image, M, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            img = cv2.warpAffine(self.image, M, size, flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

        def wm(m):
            if m is None:
                return None
            return cv2.warpAffine(m, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        pts = {k: tuple((M @ np.array([x, y, 1.0])).tolist()) for k, (x, y) in self.points.items()}
        f = self.copy()
        f.image, f.points, f.person_mask, f.hair_mask = img, pts, wm(self.person_mask), wm(self.hair_mask)
        return f


# ---------------------------------------------------------------- 단계

class Step:
    """보정 단계 하나 (steps/ 안의 파일 하나).

    - id      : 고유 이름 (설정의 켜고 끄기에 씀)
    - title   : 화면 이름
    - param_keys : 이 단계 결과에 영향을 주는 파라미터 키. 이 값이 바뀌면 이 단계부터 다시 계산
    - cache   : True면 결과 Frame을 저장 (무거운 기하 변환 뒤 등)
    - optional: 전문가 모드에서 끌 수 있는지
    """
    id: str = ""
    title: str = ""
    param_keys: tuple[str, ...] = ()
    cache: bool = False
    optional: bool = True

    def run(self, frame: Frame, analysis: Analysis, params: "RenderParams") -> Frame:  # pragma: no cover
        raise NotImplementedError


@dataclass
class RenderParams:
    """render에 들어가는 값. 단계는 param_keys로 필요한 값만 읽는다."""
    values: dict[str, Any] = field(default_factory=dict)
    disabled: frozenset[str] = frozenset()

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def digest(self, keys: tuple[str, ...]) -> str:
        sub = {k: self.values.get(k) for k in keys}
        raw = json.dumps(sub, sort_keys=True, ensure_ascii=False, default=_json_default)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _json_default(o: Any) -> Any:
    if hasattr(o, "__dict__"):
        return o.__dict__
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


@dataclass
class RenderResult:
    frame: Frame
    scale: float
    generation: int
    timings: dict[str, float]


# ---------------------------------------------------------------- 엔진

class Pipeline:
    def __init__(self, steps: list[Step]):
        self.steps = steps
        self._lock = threading.Lock()
        self._render_locks = {"preview": threading.Lock(), "full": threading.Lock()}
        self._generation = 0
        # scale별 캐시: [(누적키, Frame)] — 단계 순서대로
        self._cache: dict[str, list[tuple[str, Frame | None]]] = {}
        self._base: dict[str, Frame] = {}
        self.analysis: Analysis | None = None

    # 세션이 바뀌거나 사진이 바뀌면 호출
    def reset(self, analysis: Analysis | None = None) -> None:
        with self._lock:
            self._generation += 1
            self._cache.clear()
            self._base.clear()
            self.analysis = analysis

    @property
    def generation(self) -> int:
        return self._generation

    def bump(self) -> int:
        """새 요청이 들어오면 세대를 올려서, 그 전 요청의 결과는 버려지게 한다."""
        with self._lock:
            self._generation += 1
            return self._generation

    def is_current(self, generation: int) -> bool:
        return generation == self._generation

    def _base_frame(self, preview: bool) -> Frame:
        a = self.analysis
        assert a is not None, "prepare가 끝나지 않았습니다"
        key = "preview" if preview else "full"
        if key in self._base:
            return self._base[key]
        w, h = a.size
        s = min(1.0, PREVIEW_MAX_SIDE / max(w, h)) if preview else 1.0
        if s < 1.0:
            size = (max(1, round(w * s)), max(1, round(h * s)))
            img = cv2.resize(a.image, size, interpolation=cv2.INTER_AREA)
            rs = lambda m: None if m is None else cv2.resize(m, size, interpolation=cv2.INTER_AREA)
            pm, hm = rs(a.person_mask), rs(a.hair_mask)
            s = size[0] / w
        else:
            img, pm, hm = a.image, a.person_mask, a.hair_mask
        pts = {k: (x * s, y * s) for k, (x, y) in a.points.items()}
        f = Frame(img, pts, pm, hm, s)
        self._base[key] = f
        return f

    def render(self, params: RenderParams, preview: bool, token: CancelToken | None = None,
               generation: int | None = None,
               progress: Callable[[str, int, int], None] | None = None) -> RenderResult:
        ckey = "preview" if preview else "full"
        with self._render_locks[ckey]:
            return self._render(params, preview, ckey, token, generation, progress)

    def _render(self, params, preview, ckey, token, generation, progress) -> RenderResult:
        token = token or CancelToken()
        gen = self._generation if generation is None else generation
        if not self.is_current(gen):
            raise Cancelled()
        frame = self._base_frame(preview)
        cache = self._cache.setdefault(ckey, [])
        timings: dict[str, float] = {}
        active = [s for s in self.steps if not (s.optional and s.id in params.disabled)]
        chains, chain = [], ""
        for step in active:
            chain = hashlib.sha1((chain + step.id + params.digest(step.param_keys)).encode()).hexdigest()
            chains.append(chain)
        # 키가 맞는 가장 뒤의 저장된 결과부터 이어서 계산 (키는 누적이므로 앞 단계도 모두 같음)
        start = 0
        for j in range(min(len(cache), len(chains)) - 1, -1, -1):
            key, saved = cache[j]
            if key == chains[j] and saved is not None:
                frame, start = saved, j + 1
                break
        del cache[start:]
        for i in range(start, len(active)):
            step = active[i]
            token.check()
            if not self.is_current(gen):
                raise Cancelled()
            if progress:
                progress(step.title, i, len(active))
            t0 = time.perf_counter()
            frame = step.run(frame.copy(), self.analysis, params)  # type: ignore[arg-type]
            timings[step.id] = time.perf_counter() - t0
            keep = step.cache or i == len(active) - 1
            cache.append((chains[i], frame if keep else None))
        if timings:
            log.info("render(%s) %s", ckey, {k: round(v * 1000) for k, v in timings.items()})
        return RenderResult(frame, frame.scale, gen, timings)
