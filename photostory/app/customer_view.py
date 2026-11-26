"""① 손님 정보 (설계서 1-1)."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QRadioButton, QScrollArea,
                               QVBoxLayout, QWidget)

from ..core.session import LEVELS, Customer, Gender, Strength
from ..core.specs import SpecRegistry
from .widgets.common import StrengthPanel


class LevelButtons(QWidget):
    """기본 모드: 약하게 / 기본 / 강하게."""
    changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.group = QButtonGroup(self)
        self.buttons: dict[str, QPushButton] = {}
        for name in LEVELS:
            b = QPushButton(name)
            b.setCheckable(True)
            b.setMinimumHeight(34)
            self.group.addButton(b)
            lay.addWidget(b)
            self.buttons[name] = b
            b.clicked.connect(lambda _=False, n=name: self.changed.emit(n))
        self.group.setExclusive(True)

    def set_level(self, level: str | None) -> None:
        if level in self.buttons:
            self.buttons[level].setChecked(True)
        else:  # 부위별로 다르게 바뀐 상태
            self.group.setExclusive(False)
            for b in self.buttons.values():
                b.setChecked(False)
            self.group.setExclusive(True)


class CustomerView(QWidget):
    next_clicked = Signal()

    def __init__(self, specs: SpecRegistry, parent=None):
        super().__init__(parent)
        self.specs = specs
        self.customer = Customer()
        self.expert = False

        outer = QVBoxLayout(self)
        title = QLabel("① 손님 정보")
        title.setObjectName("title")
        outer.addWidget(title)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.name = QLineEdit()
        self.name.setPlaceholderText("예: 홍길동")
        self.name.returnPressed.connect(self._next)
        form.addRow("이름", self.name)

        self.purposes = QListWidget()
        self.purposes.setMinimumHeight(180)
        form.addRow("용도 (여러 개 선택)", self.purposes)

        self.gender_f, self.gender_m = QRadioButton("여"), QRadioButton("남")
        g = QHBoxLayout()
        g.addWidget(self.gender_f)
        g.addWidget(self.gender_m)
        g.addStretch()
        self._gender_group = QButtonGroup(self)
        self._gender_group.addButton(self.gender_f)
        self._gender_group.addButton(self.gender_m)
        form.addRow("성별", g)

        self.glasses_no, self.glasses_yes = QRadioButton("아니오"), QRadioButton("예 (사진 2장)")
        gl = QHBoxLayout()
        gl.addWidget(self.glasses_no)
        gl.addWidget(self.glasses_yes)
        gl.addStretch()
        self._glasses_group = QButtonGroup(self)
        self._glasses_group.addButton(self.glasses_no)
        self._glasses_group.addButton(self.glasses_yes)
        form.addRow("안경 착용", gl)

        self.levels = LevelButtons()
        self.levels.changed.connect(self._on_level)
        form.addRow("보정 강도", self.levels)
        outer.addLayout(form)

        self.expert_box = QScrollArea()
        self.expert_box.setWidgetResizable(True)
        self.panel = StrengthPanel(self.customer.strength)
        self.panel.changed.connect(lambda _final: self.levels.set_level(self.customer.strength.level))
        self.expert_box.setWidget(self.panel)
        self.expert_label = QLabel("전문가 모드: 부위별 강도")
        outer.addWidget(self.expert_label)
        outer.addWidget(self.expert_box, 1)

        btns = QHBoxLayout()
        btns.addStretch()
        self.next_btn = QPushButton("다음 →")
        self.next_btn.setObjectName("primary")
        self.next_btn.setMinimumHeight(40)
        self.next_btn.clicked.connect(self._next)
        btns.addWidget(self.next_btn)
        outer.addLayout(btns)

    # ---- 바깥에서 호출
    def reset(self, default_strength: Strength, expert: bool, last_purposes: list[str]) -> None:
        self.customer = Customer(strength=default_strength.copy())
        self.expert = expert
        self.name.clear()
        self.reload_specs(last_purposes)
        self.gender_f.setChecked(True)
        self.glasses_no.setChecked(True)
        self.panel.bind(self.customer.strength)
        self.levels.set_level(self.customer.strength.level)
        self.expert_box.setVisible(expert)
        self.expert_label.setVisible(expert)
        self.name.setFocus()

    def reload_specs(self, checked: list[str] | None = None) -> None:
        checked = set(checked if checked is not None else self.selected_purposes())
        self.purposes.clear()
        for s in self.specs.all():
            w, h = s.size_cm
            it = QListWidgetItem(f"{s.name}  ({w:g}×{h:g}cm{'' if s.is_white else ', 배경 ' + s.background})")
            it.setData(Qt.ItemDataRole.UserRole, s.id)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if s.id in checked else Qt.CheckState.Unchecked)
            self.purposes.addItem(it)

    def selected_purposes(self) -> list[str]:
        out = []
        for i in range(self.purposes.count()):
            it = self.purposes.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                out.append(it.data(Qt.ItemDataRole.UserRole))
        return out

    def collect(self) -> Customer:
        c = self.customer
        c.name = self.name.text().strip()
        c.purposes = self.selected_purposes()
        c.gender = Gender.MALE if self.gender_m.isChecked() else Gender.FEMALE
        c.glasses = self.glasses_yes.isChecked()
        return c

    # ---- 내부
    def _on_level(self, level: str) -> None:
        self.customer.strength.set_all(level)
        self.panel.bind(self.customer.strength)

    def _next(self) -> None:
        c = self.collect()
        errs = c.validate()
        if errs:
            QMessageBox.warning(self, "손님 정보", "\n".join(errs))
            return
        self.next_clicked.emit()
