"""④ 결과 화면 (설계서 1절 ④, 2-3, 2-4)."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from ..core.session import Session, Stroke
from ..core.specs import Spec
from .widgets.common import BrushImageView, FineSlider, ImageView, StrengthPanel

GREEN, ORANGE = "#2e7d32", "#e65100"
C_CROWN, C_CHIN, C_CENTER, C_EYE = QColor(0, 160, 255), QColor(255, 60, 60), QColor(0, 200, 120), QColor(255, 170, 0)
C_GUIDE_SOFT, C_TEMPLATE_V = QColor(255, 170, 0, 140), QColor(200, 0, 160)


class PlacementPanel(QWidget):
    """증명사진 배치 미세 조정 (설계서 2-4). 규격마다 따로."""
    changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session: Session | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.spec_box = QComboBox()
        self.spec_box.currentIndexChanged.connect(self._load)
        lay.addWidget(self.spec_box)
        self.size = FineSlider("얼굴 크기", 0, -10, 10, ticks=[0])
        self.shift = FineSlider("위아래", 0, -50, 50, ticks=[0])   # 0.1% 단위
        self.chin = FineSlider("턱 위치", 0, 0, 100, ticks=[0, 100])
        self.size.spin.setSuffix("%")
        self.shift.spin.setSuffix("‰")
        self.chin.spin.setSuffix("")
        self.chin.setToolTip("0 = 턱 최소선, 100 = 턱 최대선 (머리 볼륨이 클수록 최대선 쪽)")
        for s in (self.size, self.shift, self.chin):
            lay.addWidget(s)
            s.valueChanged.connect(lambda _v: self._save(False))
            s.committed.connect(lambda _v: self._save(True))

    def bind(self, session: Session, specs: list[Spec]) -> None:
        self.session = session
        self.spec_box.blockSignals(True)
        self.spec_box.clear()
        for s in specs:
            self.spec_box.addItem(s.name, s.id)
        self.spec_box.blockSignals(False)
        self._load()

    def _sid(self) -> str | None:
        return self.spec_box.currentData()

    def _load(self) -> None:
        if not self.session or not self._sid():
            return
        pl = self.session.placement(self._sid())
        self.size.setValue(round(pl.scale * 100))
        self.shift.setValue(round(pl.shift * 1000))
        self.chin.setValue(round(pl.chin_bias * 100))

    def _save(self, final: bool) -> None:
        if not self.session or not self._sid():
            return
        pl = self.session.placement(self._sid())
        pl.scale = self.size.value() / 100
        pl.shift = self.shift.value() / 1000
        pl.chin_bias = self.chin.value() / 100
        self.changed.emit(final)


class ResultView(QWidget):
    params_changed = Signal(bool)    # True = 손을 뗌 (원본 해상도까지)
    export_clicked = Signal()
    next_customer = Signal()
    back_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.session: Session | None = None
        self.specs: list[Spec] = []
        self._last = None

        outer = QVBoxLayout(self)
        top = QHBoxLayout()
        self.title = QLabel("④ 결과")
        self.title.setObjectName("title")
        self.status = QLabel("")
        self.guides_cb = QCheckBox("안내선 보기")
        self.guides_cb.toggled.connect(lambda _c: self._draw_guides())
        top.addWidget(self.title)
        top.addStretch()
        top.addWidget(self.status)
        top.addWidget(self.guides_cb)
        outer.addLayout(top)

        split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        cmp_row = QHBoxLayout()
        self.orig_view = ImageView("원본")
        self.ret_view = BrushImageView("수정본")
        self.ret_view.stroke_done.connect(self._on_stroke)
        for name, v in (("원본", self.orig_view), ("수정본", self.ret_view)):
            box = QVBoxLayout()
            box.addWidget(QLabel(name), 0, Qt.AlignmentFlag.AlignHCenter)
            box.addWidget(v, 1)
            cmp_row.addLayout(box, 1)
        ll.addLayout(cmp_row, 3)
        ll.addWidget(self._build_tools())
        self.tabs = QTabWidget()
        ll.addWidget(self.tabs, 2)
        self.fit = QLabel("")
        self.fit.setTextFormat(Qt.TextFormat.RichText)
        self.notes = QLabel("")
        self.notes.setWordWrap(True)
        self.notes.setStyleSheet("color:#666")
        ll.addWidget(self.fit)
        ll.addWidget(self.notes)
        split.addWidget(left)

        self.expert = QScrollArea()
        self.expert.setWidgetResizable(True)
        ew = QWidget()
        el = QVBoxLayout(ew)
        g1 = QGroupBox("부위별 강도")
        g1l = QVBoxLayout(g1)
        self.strength_panel: StrengthPanel | None = None
        self._strength_holder = g1l
        g2 = QGroupBox("증명사진 배치 미세 조정")
        g2l = QVBoxLayout(g2)
        self.placement = PlacementPanel()
        self.placement.changed.connect(self.params_changed)
        g2l.addWidget(self.placement)
        el.addWidget(g1)
        el.addWidget(g2)
        el.addStretch()
        self.expert.setWidget(ew)
        self.expert.setMinimumWidth(360)
        split.addWidget(self.expert)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        outer.addWidget(split, 1)

        btns = QHBoxLayout()
        back = QPushButton("← 사진 다시 고르기")
        back.clicked.connect(self.back_clicked)
        self.export_btn = QPushButton("출력")
        self.export_btn.setObjectName("primary")
        self.export_btn.setMinimumHeight(40)
        self.export_btn.clicked.connect(self.export_clicked)
        nxt = QPushButton("다음 손님 →")
        nxt.setMinimumHeight(40)
        nxt.clicked.connect(self.next_customer)
        btns.addWidget(back)
        btns.addStretch()
        btns.addWidget(self.export_btn)
        btns.addWidget(nxt)
        outer.addLayout(btns)

        self.transfer_views: dict[str, ImageView] = {}
        self.a4_view = ImageView("A4")

    # ---------------------------------------------------------------- 손질 도구
    def _build_tools(self) -> QWidget:
        """쌍꺼풀 기준 눈 (가이드 §5-3) · 안경 복원 붓 (v1.52 '안경수정')."""
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        self.crease_label = QLabel("쌍꺼풀 기준 눈")
        self.crease_box = QComboBox()
        for text, val in (("자동 (더 또렷한 눈)", "auto"), ("사진 왼쪽 눈", "왼눈"), ("사진 오른쪽 눈", "오른눈")):
            self.crease_box.addItem(text, val)
        self.crease_box.setToolTip("좌우 쌍꺼풀 높이가 다를 때 어느 눈에 맞출지 고릅니다")
        self.crease_box.currentIndexChanged.connect(self._on_crease)
        self.brush_cb = QCheckBox("안경 복원 붓")
        self.brush_cb.setToolTip("수정본에 칠합니다. 왼쪽 버튼 = 안경 쓴 사진에서 살리기 (못 붙은 테·힌지), "
                                 "오른쪽 버튼 = 원본으로 되돌리기 (딸려 온 머리카락 등)")
        self.brush_cb.toggled.connect(self._on_brush_toggle)
        self.brush_size = FineSlider("크기", 12, 2, 60, ticks=[12])
        self.brush_size.spin.setSuffix("")
        self.brush_size.setToolTip("붓 반지름 (사진 폭의 천분율)")
        self.brush_size.valueChanged.connect(lambda v: setattr(self.ret_view, "radius", v / 1000))
        self.swap_cb = QCheckBox("버튼 바꾸기")
        self.swap_cb.setToolTip("오른쪽 버튼이 없는 펜: 체크하면 왼쪽 버튼이 '원본으로 되돌리기'")
        self.swap_cb.toggled.connect(lambda c: setattr(self.ret_view, "swap_buttons", c))
        self.undo_btn = QPushButton("한 획 되돌리기")
        self.undo_btn.clicked.connect(self._undo_stroke)
        self.clear_btn = QPushButton("복원한 곳 모두 지우기")
        self.clear_btn.clicked.connect(self._clear_strokes)
        self.glasses_tools = [self.brush_cb, self.brush_size, self.swap_cb, self.undo_btn, self.clear_btn]
        for x in (self.crease_label, self.crease_box):
            row.addWidget(x)
        row.addSpacing(20)
        for x in self.glasses_tools:
            row.addWidget(x)
        row.addStretch()
        return w

    def _on_crease(self, _i: int) -> None:
        if self.session is None:
            return
        val = self.crease_box.currentData()
        if val != self.session.crease_ref:
            self.session.crease_ref = val
            self.params_changed.emit(True)

    def _on_brush_toggle(self, on: bool) -> None:
        self.ret_view.set_brush(on)
        if on:
            self.guides_cb.setChecked(False)

    def _on_stroke(self, mode: str, radius: float, pts: list) -> None:
        if self.session is None:
            return
        self.session.glasses_strokes.append(Stroke(mode, radius, list(pts)))
        self._update_tools()
        self.params_changed.emit(True)

    def _undo_stroke(self) -> None:
        if self.session and self.session.glasses_strokes:
            self.session.glasses_strokes.pop()
            self._update_tools()
            self.params_changed.emit(True)

    def _clear_strokes(self) -> None:
        if self.session and self.session.glasses_strokes:
            self.session.glasses_strokes.clear()
            self._update_tools()
            self.params_changed.emit(True)

    def _update_tools(self, crease=None) -> None:
        s = self.session
        glasses = bool(s and s.customer.glasses)
        for x in self.glasses_tools:
            x.setVisible(glasses)
        n = len(s.glasses_strokes) if s else 0
        self.undo_btn.setEnabled(n > 0)
        self.clear_btn.setEnabled(n > 0)
        if not glasses and self.brush_cb.isChecked():
            self.brush_cb.setChecked(False)
        if crease is not None:
            # 두 눈 모두 쌍꺼풀을 찾았을 때만 고를 필요가 있다
            both = all(crease.get(e) for e in ("왼눈", "오른눈"))
            self.crease_label.setVisible(both)
            self.crease_box.setVisible(both)

    # ---------------------------------------------------------------- 설정
    def bind(self, session: Session, specs: list[Spec], expert: bool) -> None:
        self.session, self.specs = session, specs
        self.title.setText(f"④ 결과 — {session.customer.name}")
        self.expert.setVisible(expert)
        if self.strength_panel is None:
            self.strength_panel = StrengthPanel(session.customer.strength)
            self.strength_panel.changed.connect(self.params_changed)
            self._strength_holder.addWidget(self.strength_panel)
        else:
            self.strength_panel.bind(session.customer.strength)
        self.placement.bind(session, specs)
        self.tabs.clear()
        self.transfer_views = {}
        for s in specs:
            v = ImageView(s.name)
            self.transfer_views[s.id] = v
            self.tabs.addTab(v, f"전송 · {s.name}")
        self.tabs.addTab(self.a4_view, "A4")
        self.orig_view.set_image(None)
        self.ret_view.set_image(None)
        self.a4_view.set_image(None)
        self.fit.clear()
        self.notes.clear()
        self._last = None
        self.crease_box.blockSignals(True)
        self.crease_box.setCurrentIndex(max(0, self.crease_box.findData(session.crease_ref)))
        self.crease_box.blockSignals(False)
        self.brush_cb.setChecked(False)
        self._update_tools(crease={})

    def set_busy(self, text: str) -> None:
        self.status.setText(text)

    def set_original(self, rgb) -> None:
        self.orig_view.set_image(rgb)

    # ---------------------------------------------------------------- 결과 표시
    def show_result(self, frame, full: bool) -> None:
        out = frame.outputs
        if not full:  # 수정본은 미리보기 해상도로 보여 준다 (원본 해상도는 출력에만 씀)
            self.ret_view.set_image(out.get("retouched"))
            self._last = frame
        for sid, img in out.get("transfers", {}).items():
            if sid in self.transfer_views:
                self.transfer_views[sid].set_image(img)
        if "a4" in out:
            self.a4_view.set_image(out["a4"])
        self._show_fit(out.get("transfer_checks", {}))
        if "crease" in out:
            self._update_tools(crease=out["crease"])
        notes = [n for n in frame.notes]
        self.notes.setText(" · ".join(notes))
        self._draw_guides()

    def _show_fit(self, checks) -> None:
        parts = []
        for s in self.specs:
            c = checks.get(s.id)
            if c is None:
                continue
            color = GREEN if c.ok else ORANGE
            parts.append(f"<span style='color:{color}'>{s.name}: 턱끝~정수리 {c.head_cm:.2f}cm "
                         f"(목표 {s.head_cm:g}cm)</span>")
        self.fit.setText(" &nbsp;|&nbsp; ".join(parts))

    def _draw_guides(self) -> None:
        on = self.guides_cb.isChecked()
        f = self._last
        if not on or f is None or "guides" not in f.outputs:
            self.ret_view.set_guides()
            for v in self.transfer_views.values():
                v.set_guides()
            return
        g = f.outputs["guides"]
        h = [(g["crown_y"], C_CROWN, "정수리")]
        v = [(g["center_x"], C_CENTER, "")]
        t = f.outputs.get("template")
        if t is not None:
            # 가이드 §5 템플릿 (여: A~K·색 이름 / 남: 1~12·A~F). 사진에서 다시 찾은 선은 진하게
            h += [(l.pos, C_EYE if l.basis == "영상" else C_GUIDE_SOFT, l.id) for l in t.h]
            for l in t.v:
                if l.id in ("중심", "A"):
                    continue
                v += [(x, C_TEMPLATE_V, l.id) for x in t.v_x(l)]
        else:
            h += [(g["chin_y"], C_CHIN, "턱"), (g["eye_y"], C_EYE, "")]
        self.ret_view.set_guides(h=h, v=v)
        for s in self.specs:
            v = self.transfer_views.get(s.id)
            if v:
                v.set_guides(h=[(s.crown_line, C_CROWN, ""), (s.chin_min_line, C_CHIN, ""),
                                (s.chin_max_line, C_CHIN, "")],
                             v=[(s.center_x, C_CENTER, "")])
