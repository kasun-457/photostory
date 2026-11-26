"""② 사진 업로드 (설계서 1-2).

- 안경 아니오 → 칸 1개, 예 → 칸 2개 (원본 + 안경 쓴 사진)
- 끌어다 놓기와 [열기]. 두 장을 한꺼번에 놓으면 이름이 `안경`으로 끝나는 파일을 안경 칸에
- 칸에 들어가는 순간 뒤에서 분석(prepare)을 시작하고, 얼굴을 못 찾으면 그 자리에서 알려 준다
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFileDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                               QWidget)

from ..core.io.imageio import SUPPORTED_EXTS, is_supported, read_thumbnail
from .widgets.common import ImageView

FILTER = "사진 (" + " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTS)) + ")"


def is_glasses_name(p: Path) -> bool:
    return p.stem.replace(" ", "").endswith("안경")


class PhotoSlot(QFrame):
    dropped = Signal(list)          # 여러 장이 놓이면 부모가 나눠 준다
    cleared = Signal()

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("slot")
        self.path: Path | None = None
        lay = QVBoxLayout(self)
        self.title = QLabel(title)
        self.title.setObjectName("slotTitle")
        self.view = ImageView("여기에 사진을 끌어다 놓으세요")
        self.status = QLabel("")
        self.status.setWordWrap(True)
        row = QHBoxLayout()
        self.open_btn = QPushButton("열기…")
        self.open_btn.clicked.connect(self._open)
        self.clear_btn = QPushButton("비우기")
        self.clear_btn.clicked.connect(self.clear)
        row.addWidget(self.open_btn)
        row.addWidget(self.clear_btn)
        row.addStretch()
        lay.addWidget(self.title)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.status)
        lay.addLayout(row)

    def set_path(self, p: Path | None, thumb=None) -> None:
        self.path = p
        self.view.set_image(thumb)
        self.set_status("" if p is None else f"{p.name} · 분석 중…", None)

    def set_status(self, text: str, ok: bool | None) -> None:
        color = {True: "#2e7d32", False: "#c62828", None: "#666"}[ok]
        self.status.setStyleSheet(f"color: {color}")
        self.status.setText(text)

    def clear(self) -> None:
        self.set_path(None)
        self.cleared.emit()

    def _open(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "사진 열기", "", FILTER)
        if files:
            self.dropped.emit([Path(f) for f in files])

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        files = [Path(u.toLocalFile()) for u in e.mimeData().urls() if u.isLocalFile()]
        files = [f for f in files if is_supported(f)]
        if files:
            self.dropped.emit(files)


class UploadView(QWidget):
    back_clicked = Signal()
    start_clicked = Signal()
    photo_set = Signal(str, object)      # ("original"|"glasses", Path|None)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        outer = QVBoxLayout(self)
        self.title = QLabel("② 사진 업로드")
        self.title.setObjectName("title")
        outer.addWidget(self.title)
        row = QHBoxLayout()
        self.original = PhotoSlot("원본 (안경 벗은 사진)")
        self.glasses = PhotoSlot("안경 쓴 사진")
        self.swap_btn = QPushButton("⇄ 칸 바꾸기")
        self.swap_btn.clicked.connect(self.swap)
        row.addWidget(self.original, 1)
        row.addWidget(self.swap_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.glasses, 1)
        outer.addLayout(row, 1)
        btns = QHBoxLayout()
        back = QPushButton("← 손님 정보")
        back.clicked.connect(self.back_clicked)
        self.start_btn = QPushButton("시작")
        self.start_btn.setObjectName("primary")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.clicked.connect(self.start_clicked)
        btns.addWidget(back)
        btns.addStretch()
        btns.addWidget(self.start_btn)
        outer.addLayout(btns)
        self.original.dropped.connect(lambda f: self.accept_files(f, "original"))
        self.glasses.dropped.connect(lambda f: self.accept_files(f, "glasses"))
        self.original.cleared.connect(lambda: self._set("original", None))
        self.glasses.cleared.connect(lambda: self._set("glasses", None))
        self.need_glasses = False
        self._ok = {"original": False, "glasses": False}

    def configure(self, name: str, need_glasses: bool) -> None:
        self.need_glasses = need_glasses
        self.title.setText(f"② 사진 업로드 — {name}")
        self.glasses.setVisible(need_glasses)
        self.swap_btn.setVisible(need_glasses)
        self.original.title.setText("원본 (안경 벗은 사진)" if need_glasses else "원본")
        self.update_start()

    def reset(self) -> None:
        self.original.set_path(None)
        self.glasses.set_path(None)
        self._ok = {"original": False, "glasses": False}
        self.update_start()

    def slot(self, which: str) -> PhotoSlot:
        return self.original if which == "original" else self.glasses

    def accept_files(self, files: list[Path], target: str) -> None:
        if not self.need_glasses or len(files) == 1:
            self._set(target, files[0])
            return
        g = [f for f in files if is_glasses_name(f)]
        o = [f for f in files if not is_glasses_name(f)]
        if g and o:
            self._set("original", o[0])
            self._set("glasses", g[0])
        else:  # 구분할 수 없으면 놓은 순서대로
            self._set("original", files[0])
            self._set("glasses", files[1])

    def _set(self, which: str, p: Path | None) -> None:
        thumb = None
        if p is not None:
            try:
                thumb = read_thumbnail(p, 600)
            except Exception as e:  # noqa: BLE001
                self.slot(which).set_path(None)
                self.slot(which).set_status(str(e), False)
                return
        self._ok[which] = False
        self.slot(which).set_path(p, thumb)
        self.update_start()
        self.photo_set.emit(which, p)

    def mark(self, which: str, path: Path, ok: bool, message: str) -> None:
        s = self.slot(which)
        if s.path != path:
            return  # 그 사이 다른 사진으로 바뀜
        self._ok[which] = ok
        s.set_status(f"{path.name} · {message}", ok)
        self.update_start()

    def swap(self) -> None:
        a, b = self.original.path, self.glasses.path
        self._set("original", b) if b else self.original.clear()
        self._set("glasses", a) if a else self.glasses.clear()

    def update_start(self) -> None:
        ready = self.original.path is not None and (not self.need_glasses or self.glasses.path is not None)
        self.start_btn.setEnabled(ready)

    # 창 전체에 놓아도 받는다
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        files = [Path(u.toLocalFile()) for u in e.mimeData().urls() if u.isLocalFile()]
        files = [f for f in files if is_supported(f)]
        if files:
            self.accept_files(files, "original")
