"""메인 창: 화면 흐름 ①→②→③→④→⑤ (설계서 1절). 화면은 표시만 하고 계산은 Processor에 맡긴다."""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMainWindow, QMessageBox, QStackedWidget

from ..core.export import Conflict, existing, planned_names, save_all
from ..core.pipeline import CancelToken, Cancelled
from ..core.processor import Processor
from ..core.session import Session, Strength
from ..core.specs import SpecRegistry
from ..models.hub import ModelHub
from ..steps.a4 import DEFAULT_FONT
from .customer_view import CustomerView
from .dialogs import OutputFolderDialog, ask_conflict
from .result_view import ResultView
from .settings import AppSettings
from .settings_view import SettingsDialog
from .upload_view import UploadView
from .workers import Runner

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self, hub: ModelHub | None = None, settings: AppSettings | None = None):
        super().__init__()
        self.setWindowTitle("사진이야기 증명사진 보정")
        self.resize(1280, 860)
        self.settings = settings or AppSettings.load()
        self.specs = SpecRegistry(self.settings.specs_dir, set(self.settings.hidden_specs))
        self.hub = hub or ModelHub()
        self.hub.preload()                       # 모델 미리 읽기 (설계서 6-1)
        self.proc = Processor(self.hub, self.specs, self._font())
        self.runner = Runner(threads=3)
        self.session = Session()
        self._token = CancelToken()
        self._full_result = None                 # (generation, RenderResult)
        self._pending_export = False
        self._preview_busy = self._preview_dirty = False
        self._full_wanted = False

        self.customer_view = CustomerView(self.specs)
        self.upload_view = UploadView()
        self.result_view = ResultView()
        self.stack = QStackedWidget()
        for w in (self.customer_view, self.upload_view, self.result_view):
            self.stack.addWidget(w)
        self.setCentralWidget(self.stack)

        act = QAction("설정", self)
        act.triggered.connect(self.open_settings)
        self.menuBar().addAction(act)

        self.customer_view.next_clicked.connect(self.to_upload)
        self.upload_view.back_clicked.connect(lambda: self.stack.setCurrentWidget(self.customer_view))
        self.upload_view.photo_set.connect(self.on_photo)
        self.upload_view.start_clicked.connect(self.start)
        self.result_view.params_changed.connect(self.on_params)
        self.result_view.export_clicked.connect(self.export)
        self.result_view.next_customer.connect(self.next_customer)
        self.result_view.back_clicked.connect(lambda: self.stack.setCurrentWidget(self.upload_view))

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(40)
        self._debounce.timeout.connect(self._run_preview)
        self.next_customer()

    # ---------------------------------------------------------------- 설정
    def _font(self) -> Path:
        p = Path(self.settings.font_path) if self.settings.font_path else DEFAULT_FONT
        return p if p.exists() else DEFAULT_FONT

    def _default_strength(self) -> Strength:
        if self.settings.expert:
            return Strength(dict(self.settings.default_parts), None)
        return Strength.from_level(self.settings.default_level)

    def open_settings(self) -> None:
        d = SettingsDialog(self.settings, self.specs, self)
        if d.exec():
            self.proc.font = self._font()
            self.session.disabled_steps = set(self.settings.disabled_steps) if self.settings.expert else set()
            self.customer_view.expert = self.settings.expert
            self.customer_view.expert_box.setVisible(self.settings.expert)
            self.customer_view.expert_label.setVisible(self.settings.expert)
            self.customer_view.reload_specs()
            if self.stack.currentWidget() is self.result_view:
                self.result_view.expert.setVisible(self.settings.expert)
                self.on_params(True)

    # ---------------------------------------------------------------- ① → ②
    def next_customer(self) -> None:
        """이전 손님의 정보와 계산 결과를 모두 지운다."""
        self._token.cancel()
        self._token = CancelToken()
        self.proc.clear()
        self.session = Session(disabled_steps=set(self.settings.disabled_steps) if self.settings.expert else set())
        self._full_result = None
        self._pending_export = False
        self.customer_view.reset(self._default_strength(), self.settings.expert, self.settings.last_purposes)
        self.upload_view.reset()
        self.stack.setCurrentWidget(self.customer_view)

    def to_upload(self) -> None:
        c = self.customer_view.collect()
        self.session.customer = c
        self.settings.last_purposes = list(c.purposes)
        self.settings.save()
        if not c.glasses and self.session.glasses_photo:
            self.upload_view.glasses.clear()
        self.upload_view.configure(c.name, c.glasses)
        self.stack.setCurrentWidget(self.upload_view)

    # ---------------------------------------------------------------- ② 업로드하면서 분석 시작
    def on_photo(self, which: str, path: Path | None) -> None:
        if which == "original":
            self.session.original = path
        else:
            self.session.glasses_photo = path
        if path is None:
            return
        token = self._token

        def work(_progress):
            return self.proc.prepare(path, token)

        def done(a):
            n = a.extra.get("face_count", 1)
            msg = "얼굴 확인됨" + (" (얼굴이 여러 개라 가장 큰 얼굴 사용)" if n > 1 else "")
            self.upload_view.mark(which, path, True, msg)

        def fail(e):
            if not isinstance(e, Cancelled):
                self.upload_view.mark(which, path, False, str(e))

        self.runner.submit(work, on_done=done, on_fail=fail)

    # ---------------------------------------------------------------- ③ 자동 처리
    def start(self) -> None:
        if not self.session.ready_to_start():
            return
        specs = self.proc.chosen_specs(self.session)
        self.result_view.bind(self.session, specs, self.settings.expert)
        self.stack.setCurrentWidget(self.result_view)
        self.result_view.set_busy("분석 중…")
        self._full_result = None
        token = self._token

        def work(progress):
            self.proc.start(self.session, token)
            gen = self.proc.pipeline.bump()
            return gen, self.proc.render(self.session, True, token, gen)

        def done(res):
            gen, r = res
            self.result_view.set_original(self.proc.original_preview())
            self._show(r, full=False)
            self._run_full(gen)

        self.runner.submit(work, on_done=done, on_fail=self._fail)

    def _fail(self, e: Exception) -> None:
        if isinstance(e, Cancelled):
            return
        self.result_view.set_busy("")
        QMessageBox.warning(self, "처리 실패", str(e))
        if self.stack.currentWidget() is self.result_view and self.proc.pipeline.analysis is None:
            self.stack.setCurrentWidget(self.upload_view)

    def _show(self, r, full: bool) -> None:
        if not self.proc.pipeline.is_current(r.generation):
            return
        self.result_view.show_result(r.frame, full)
        if full:
            self.result_view.set_busy("")
        else:
            self.result_view.set_busy("원본 해상도 계산 중…")

    # ---------------------------------------------------------------- ④ 강도·배치 바꾸면 다시 계산
    def on_params(self, final: bool) -> None:
        self._full_wanted = self._full_wanted or final
        self._debounce.start()

    def _run_preview(self) -> None:
        if self._preview_busy:
            self._preview_dirty = True
            return
        self._preview_busy = True
        gen = self.proc.pipeline.bump()
        self._full_result = None
        want_full, self._full_wanted = self._full_wanted, False
        token = self._token

        def work(_p):
            return self.proc.render(self.session, True, token, gen)

        def done(r):
            self._preview_busy = False
            self._show(r, full=False)
            if self._preview_dirty:
                self._preview_dirty = False
                self._full_wanted = self._full_wanted or want_full
                self._run_preview()
            elif want_full or self._pending_export:
                self._run_full(gen)

        def fail(e):
            self._preview_busy = False
            if self._preview_dirty:
                self._preview_dirty = False
                self._run_preview()
            else:
                self._fail(e)

        self.runner.submit(work, on_done=done, on_fail=fail)

    def _run_full(self, gen: int) -> None:
        token = self._token
        self.result_view.set_busy("원본 해상도 계산 중…")

        def work(_p):
            return self.proc.render(self.session, False, token, gen)

        def done(r):
            if not self.proc.pipeline.is_current(r.generation):
                return
            self._full_result = (r.generation, r)
            self._show(r, full=True)
            if self._pending_export:
                self._pending_export = False
                self._save(r)

        self.runner.submit(work, on_done=done, on_fail=self._fail)

    # ---------------------------------------------------------------- ⑤ 출력
    def export(self) -> None:
        s = self.settings
        specs = self.proc.chosen_specs(self.session)
        names = [n for _, n in planned_names(self.session.customer.name, specs)]
        folder = Path(s.output_dir) if s.output_dir else None
        if not (s.fixed_output and folder and folder.is_dir()):
            d = OutputFolderDialog(s.output_dir, s.fixed_output, names, self)
            if not d.exec():
                return
            folder = d.folder
            s.output_dir, s.fixed_output = str(folder), d.remember.isChecked()
            s.save()
        self._export_folder = folder
        fr = self._full_result
        if fr and self.proc.pipeline.is_current(fr[0]):
            self._save(fr[1])
        else:
            self._pending_export = True
            self.result_view.set_busy("원본 해상도 계산이 끝나면 저장합니다…")
            if not self._preview_busy:
                self._run_full(self.proc.pipeline.generation)

    def _save(self, r) -> None:
        folder = self._export_folder
        specs = self.proc.chosen_specs(self.session)
        name = self.session.customer.name
        names = [n for _, n in planned_names(name, specs)]
        clash = existing(folder, names)
        choice = Conflict.OVERWRITE
        if clash:
            choice = ask_conflict(self, clash)
            if choice == Conflict.CANCEL:
                self.result_view.set_busy("")
                return
        try:
            saved = save_all(folder, name, specs, r.frame.outputs, choice)
        except OSError as e:
            QMessageBox.warning(self, "출력 실패", str(e))
            return
        msg = f"저장 완료 ({folder}): " + ", ".join(p.name for p in saved or [])
        self.result_view.set_busy("저장 완료")
        self.statusBar().showMessage(msg, 15000)

    def closeEvent(self, e):
        self._token.cancel()
        self.runner.wait(3000)
        self.hub.close()
        super().closeEvent(e)
