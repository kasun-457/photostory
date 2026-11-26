"""뒤에서 도는 작업. 화면은 멈추지 않고, 늦게 끝난 결과는 세대 번호로 버린다 (설계서 6-1)."""
from __future__ import annotations

import logging
import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

log = logging.getLogger(__name__)


class _Signals(QObject):
    done = Signal(object, object)       # (tag, result)
    failed = Signal(object, object)     # (tag, exception)
    progress = Signal(object, str)      # (tag, message)


class Task(QRunnable):
    def __init__(self, fn: Callable[..., Any], tag: Any = None):
        super().__init__()
        self.fn, self.tag = fn, tag
        self.signals = _Signals()
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            res = self.fn(lambda msg: self.signals.progress.emit(self.tag, msg))
        except Exception as e:  # noqa: BLE001 - 화면에 알려 준다
            if type(e).__name__ != "Cancelled":
                log.error("작업 실패 %s\n%s", self.tag, traceback.format_exc())
            self.signals.failed.emit(self.tag, e)
            return
        self.signals.done.emit(self.tag, res)


class _Relay(QObject):
    """GUI 스레드에 사는 받는 쪽. 작업 스레드의 신호를 GUI 스레드에서 콜백으로 넘긴다."""

    def __init__(self, on_done, on_fail, on_progress, parent=None):
        super().__init__(parent)
        self._d, self._f, self._p = on_done, on_fail, on_progress

    @Slot(object, object)
    def done(self, tag, r):
        if self._d:
            self._d(r)
        self.deleteLater()

    @Slot(object, object)
    def failed(self, tag, e):
        if self._f:
            self._f(e)
        self.deleteLater()

    @Slot(object, str)
    def progress(self, tag, m):
        if self._p:
            self._p(m)


class Runner:
    """화면 쪽에서 쓰는 작은 도우미. 결과 콜백은 GUI 스레드에서 불린다."""

    def __init__(self, threads: int = 2):
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(threads)
        self._relays: set[_Relay] = set()   # 파이썬 쪽에서 지워지지 않도록 붙잡아 둔다

    def submit(self, fn, tag=None, on_done=None, on_fail=None, on_progress=None) -> Task:
        t = Task(fn, tag)
        relay = _Relay(on_done, on_fail, on_progress)
        self._relays.add(relay)
        relay.destroyed.connect(lambda *_: self._relays.discard(relay))
        t.signals.done.connect(relay.done)
        t.signals.failed.connect(relay.failed)
        t.signals.progress.connect(relay.progress)
        self.pool.start(t)
        return t

    def wait(self, ms: int = -1) -> bool:
        return self.pool.waitForDone(ms)
