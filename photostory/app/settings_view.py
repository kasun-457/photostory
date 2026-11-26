"""설정 화면 (설계서 5절)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSpinBox,
                               QTabWidget, QVBoxLayout, QWidget)

from ..core.session import LEVELS, Strength
from ..core.specs import Spec, SpecRegistry
from ..steps import default_steps
from .settings import AppSettings
from .widgets.common import StrengthPanel


class SpecEditor(QDialog):
    def __init__(self, spec: Spec, is_new: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("규격 추가" if is_new else f"규격 수정 — {spec.name}")
        self.spec = spec
        f = QFormLayout(self)
        self.sid = QLineEdit(spec.id)
        self.sid.setEnabled(is_new)
        self.name = QLineEdit(spec.name)
        self.label = QLineEdit(spec.a4_label)
        self.file_label = QLineEdit(spec.file_label)

        def dspin(v, lo, hi, dec=2):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setSingleStep(0.01)
            s.setValue(v)
            return s

        self.w = dspin(spec.size_cm[0], 0.5, 20)
        self.h = dspin(spec.size_cm[1], 0.5, 20)
        self.head = dspin(spec.head_cm, 0.5, 15)
        self.top = dspin(spec.top_cm, 0, 10)
        self.margin = dspin(spec.chin_margin_cm, 0, 2)
        self.dpi = QSpinBox()
        self.dpi.setRange(72, 1200)
        self.dpi.setValue(int(spec.dpi))
        self.bg = QLineEdit(spec.background)
        pick = QPushButton("색 고르기…")
        pick.clicked.connect(self._pick)
        bgrow = QHBoxLayout()
        bgrow.addWidget(self.bg)
        bgrow.addWidget(pick)
        self.a4 = QSpinBox()
        self.a4.setRange(0, 20)
        self.a4.setSpecialValueText("자동 (공식)")
        self.a4.setValue(spec.a4_count or 0)
        self.aliases = QLineEdit(", ".join(spec.aliases))
        size = QHBoxLayout()
        size.addWidget(self.w)
        size.addWidget(QLabel("×"))
        size.addWidget(self.h)
        size.addWidget(QLabel("cm"))
        f.addRow("id (영문)", self.sid)
        f.addRow("이름", self.name)
        f.addRow("A4 라벨", self.label)
        f.addRow("파일 이름에 쓸 이름", self.file_label)
        f.addRow("크기", size)
        f.addRow("전송 dpi", self.dpi)
        f.addRow("정수리~턱 (cm)", self.head)
        f.addRow("머리 윗공간 (cm)", self.top)
        f.addRow("턱 최대선 여유 (cm)", self.margin)
        f.addRow("배경색", bgrow)
        f.addRow("A4 장수", self.a4)
        f.addRow("별칭 (쉼표로)", self.aliases)
        if spec.override_px:
            f.addRow("", QLabel("이 규격은 실제 템플릿 px 값을 씁니다 (가이드 §15-1)"))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def _pick(self) -> None:
        from PySide6.QtGui import QColor
        c = QColorDialog.getColor(QColor(self.bg.text()), self)
        if c.isValid():
            self.bg.setText(c.name())

    def result_spec(self) -> Spec:
        s = Spec.from_json({**self.spec.to_json(),
                            "id": self.sid.text().strip(), "name": self.name.text().strip(),
                            "a4_label": self.label.text().strip(), "file_label": self.file_label.text().strip(),
                            "size_cm": [self.w.value(), self.h.value()], "dpi": self.dpi.value(),
                            "head_cm": self.head.value(), "top_cm": self.top.value(),
                            "chin_margin_cm": self.margin.value(), "background": self.bg.text().strip(),
                            "a4_count": self.a4.value() or None,
                            "aliases": [a.strip() for a in self.aliases.text().split(",") if a.strip()]})
        return s

    def _ok(self) -> None:
        errs = self.result_spec().validate()
        if errs:
            QMessageBox.warning(self, "규격", "\n".join(errs))
            return
        self.accept()


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, specs: SpecRegistry, parent=None):
        super().__init__(parent)
        self.setWindowTitle("설정")
        self.settings, self.specs = settings, specs
        self.resize(640, 560)
        tabs = QTabWidget()

        # 일반
        gen = QWidget()
        gf = QFormLayout(gen)
        self.mode = QComboBox()
        self.mode.addItems(["기본 모드", "전문가 모드"])
        self.mode.setCurrentIndex(1 if settings.expert else 0)
        self.level = QComboBox()
        self.level.addItems(list(LEVELS))
        self.level.setCurrentText(settings.default_level)
        self.default_strength = Strength(dict(settings.default_parts), None)
        self.parts = StrengthPanel(self.default_strength)
        self.font = QLineEdit(settings.font_path)
        self.font.setPlaceholderText("비우면 나눔고딕 Bold")
        fb = QPushButton("찾아보기…")
        fb.clicked.connect(self._font)
        frow = QHBoxLayout()
        frow.addWidget(self.font)
        frow.addWidget(fb)
        gf.addRow("모드", self.mode)
        gf.addRow("기본 보정 강도 (기본 모드)", self.level)
        gf.addRow("기본 부위별 강도 (전문가 모드)", self.parts)
        gf.addRow("A4 라벨 글꼴", frow)
        gf.addRow("", QLabel("무료 오픈 글꼴(.ttf/.otf)을 고르세요."))
        tabs.addTab(gen, "일반")

        # 출력
        out = QWidget()
        of = QFormLayout(out)
        self.fixed = QCheckBox("고정 위치 사용 (출력할 때 묻지 않음)")
        self.fixed.setChecked(settings.fixed_output)
        self.outdir = QLineEdit(settings.output_dir)
        ob = QPushButton("바꾸기…")
        ob.clicked.connect(self._outdir)
        orow = QHBoxLayout()
        orow.addWidget(self.outdir)
        orow.addWidget(ob)
        of.addRow(self.fixed)
        of.addRow("출력 폴더", orow)
        tabs.addTab(out, "출력")

        # 규격
        sp = QWidget()
        sl = QVBoxLayout(sp)
        self.spec_list = QListWidget()
        sl.addWidget(self.spec_list, 1)
        row = QHBoxLayout()
        for text, fn in (("추가", self._add), ("수정", self._edit), ("숨기기/삭제", self._hide),
                         ("다시 보이기·기본값으로", self._reset)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        sl.addLayout(row)
        tabs.addTab(sp, "규격 관리")

        # 단계
        st = QWidget()
        stl = QVBoxLayout(st)
        stl.addWidget(QLabel("전문가 모드에서만 쓰입니다. 꺼진 단계는 출력에서도 빠집니다."))
        self.step_boxes: dict[str, QCheckBox] = {}
        for s in default_steps():
            cb = QCheckBox(s.title)
            cb.setChecked(s.id not in settings.disabled_steps)
            cb.setEnabled(s.optional)
            stl.addWidget(cb)
            self.step_boxes[s.id] = cb
        stl.addStretch()
        self.steps_tab = tabs.addTab(st, "단계 켜고 끄기")
        tabs.setTabVisible(self.steps_tab, settings.expert)
        self.mode.currentIndexChanged.connect(lambda i: tabs.setTabVisible(self.steps_tab, i == 1))

        lay = QVBoxLayout(self)
        lay.addWidget(tabs)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._fill_specs()

    def _fill_specs(self) -> None:
        self.spec_list.clear()
        for s in self.specs.all(include_hidden=True):
            hidden = s.id in self.specs.hidden
            tag = "기본" if s.builtin else "사용자"
            it = QListWidgetItem(f"{s.name}  [{tag}]{'  (숨김)' if hidden else ''}")
            it.setData(Qt.ItemDataRole.UserRole, s.id)
            if hidden:
                it.setForeground(Qt.GlobalColor.gray)
            self.spec_list.addItem(it)

    def _current(self) -> str | None:
        it = self.spec_list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it else None

    def _add(self) -> None:
        base = Spec(id="custom", name="", size_cm=[3.5, 4.5], head_cm=3.2, top_cm=0.2)
        d = SpecEditor(base, True, self)
        if d.exec():
            try:
                self.specs.save(d.result_spec())
            except (ValueError, RuntimeError) as e:
                QMessageBox.warning(self, "규격", str(e))
            self._fill_specs()

    def _edit(self) -> None:
        sid = self._current()
        if not sid:
            return
        d = SpecEditor(self.specs.get(sid), False, self)
        if d.exec():
            try:
                self.specs.save(d.result_spec())
            except (ValueError, RuntimeError) as e:
                QMessageBox.warning(self, "규격", str(e))
            self._fill_specs()

    def _hide(self) -> None:
        sid = self._current()
        if sid:
            self.specs.delete(sid)
            self._fill_specs()

    def _reset(self) -> None:
        sid = self._current()
        if sid:
            self.specs.reset_builtin(sid)
            self._fill_specs()

    def _font(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "글꼴", "", "글꼴 (*.ttf *.otf *.ttc)")
        if f:
            self.font.setText(f)

    def _outdir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "출력 폴더", self.outdir.text())
        if d:
            self.outdir.setText(d)

    def _save(self) -> None:
        s = self.settings
        s.expert = self.mode.currentIndex() == 1
        s.default_level = self.level.currentText()
        s.default_parts = dict(self.default_strength.values)
        font = self.font.text().strip()
        if font and not Path(font).exists():
            QMessageBox.warning(self, "설정", "글꼴 파일을 찾을 수 없습니다")
            return
        s.font_path = font
        s.fixed_output = self.fixed.isChecked()
        s.output_dir = self.outdir.text().strip()
        s.hidden_specs = sorted(self.specs.hidden)
        s.disabled_steps = [k for k, cb in self.step_boxes.items() if not cb.isChecked()]
        s.save()
        self.accept()
