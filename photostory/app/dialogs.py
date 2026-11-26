"""⑤ 출력 대화상자와 같은 이름 처리 (설계서 1절 ⑤, 4절)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget)

from ..core.export import Conflict


class OutputFolderDialog(QDialog):
    def __init__(self, folder: str, remember: bool, files: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("출력")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("저장할 폴더"))
        row = QHBoxLayout()
        self.path = QLineEdit(folder)
        browse = QPushButton("찾아보기…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.path, 1)
        row.addWidget(browse)
        lay.addLayout(row)
        self.remember = QCheckBox("이 위치 계속 사용 (다음부터 묻지 않음, 설정에서 바꿀 수 있음)")
        self.remember.setChecked(remember)
        lay.addWidget(self.remember)
        lay.addWidget(QLabel("저장할 파일:\n" + "\n".join(f"  · {f}" for f in files)))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Save).setText("저장")
        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(560, 0)

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "출력 폴더", self.path.text())
        if d:
            self.path.setText(d)

    def _accept(self) -> None:
        if not self.path.text().strip():
            QMessageBox.warning(self, "출력", "폴더를 고르세요")
            return
        self.accept()

    @property
    def folder(self) -> Path:
        return Path(self.path.text().strip())


def ask_conflict(parent: QWidget, names: list[str]) -> Conflict:
    box = QMessageBox(parent)
    box.setWindowTitle("같은 이름 파일")
    box.setText("같은 이름의 파일이 이미 있습니다:\n" + "\n".join(names))
    over = box.addButton(Conflict.OVERWRITE.value, QMessageBox.ButtonRole.DestructiveRole)
    num = box.addButton(Conflict.NUMBER.value, QMessageBox.ButtonRole.AcceptRole)
    box.addButton(Conflict.CANCEL.value, QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(num)
    box.exec()
    if box.clickedButton() is over:
        return Conflict.OVERWRITE
    if box.clickedButton() is num:
        return Conflict.NUMBER
    return Conflict.CANCEL
