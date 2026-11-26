"""공용 위젯: 이미지 표시, 강도 슬라이더, 부위별 강도 판."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider,
                               QSpinBox, QWidget)

from ...core.session import LEVELS, PARTS, Strength


def to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class ImageView(QWidget):
    """비율을 지키며 칸에 맞춰 그린다. 안내선(가로·세로 선)을 겹쳐 그릴 수 있다."""

    def __init__(self, placeholder: str = "", parent=None):
        super().__init__(parent)
        self._pix: QPixmap | None = None
        self._img_size = (0, 0)
        self._hlines: list[tuple[float, QColor, str]] = []
        self._vlines: list[tuple[float, QColor, str]] = []
        self.placeholder = placeholder
        self.setMinimumSize(120, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, rgb: np.ndarray | None) -> None:
        if rgb is None:
            self._pix = None
            self._img_size = (0, 0)
        else:
            self._pix = QPixmap.fromImage(to_qimage(rgb))
            self._img_size = (rgb.shape[1], rgb.shape[0])
        self.update()

    def set_guides(self, h: list[tuple[float, QColor, str]] | None = None,
                   v: list[tuple[float, QColor, str]] | None = None) -> None:
        """이미지 px 좌표 기준 가로선(y)·세로선(x)."""
        self._hlines, self._vlines = h or [], v or []
        self.update()

    def _target(self) -> QRectF:
        w, h = self._img_size
        if not w:
            return QRectF()
        s = min(self.width() / w, self.height() / h)
        tw, th = w * s, h * s
        return QRectF((self.width() - tw) / 2, (self.height() - th) / 2, tw, th)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self._pix is None:
            p.setPen(QColor(140, 140, 140))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)
            return
        r = self._target()
        p.drawPixmap(r, self._pix, QRectF(self._pix.rect()))
        s = r.width() / self._img_size[0]
        for y, c, label in self._hlines:
            p.setPen(QPen(c, 1))
            yy = r.top() + y * s
            p.drawLine(int(r.left()), int(yy), int(r.right()), int(yy))
            if label:
                p.drawText(int(r.left()) + 4, int(yy) - 3, label)
        for x, c, label in self._vlines:
            p.setPen(QPen(c, 1))
            xx = r.left() + x * s
            p.drawLine(int(xx), int(r.top()), int(xx), int(r.bottom()))
            if label:
                p.drawText(int(xx) + 3, int(r.bottom()) - 4, label)


class BrushImageView(ImageView):
    """칠할 수 있는 이미지 칸 (안경 복원 붓). 왼쪽 버튼 = "glasses", 오른쪽 버튼 = "original".

    획은 이미지 크기 대비 비율 좌표로 내보낸다 (미리보기·원본 해상도가 같은 곳을 칠하도록).
    radius도 이미지 폭 대비 비율이다.
    """
    stroke_done = Signal(str, float, list)   # mode, radius, [(x, y)]
    COLORS = {"glasses": QColor(0, 150, 255, 90), "original": QColor(255, 80, 0, 90)}

    def __init__(self, placeholder: str = "", parent=None):
        super().__init__(placeholder, parent)
        self.brush_on = False
        self.radius = 0.012
        self.swap_buttons = False          # 오른쪽 버튼이 없는 펜용 (v1.52 Alt+Y)
        self._mode: str | None = None
        self._pts: list[tuple[float, float]] = []
        self._hover: QPointF | None = None
        self.setMouseTracking(True)

    def set_brush(self, on: bool) -> None:
        self.brush_on = on
        self._mode, self._pts = None, []
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)
        self.update()

    def _to_img(self, pos) -> tuple[float, float] | None:
        r = self._target()
        if r.isEmpty():
            return None
        x, y = (pos.x() - r.left()) / r.width(), (pos.y() - r.top()) / r.height()
        return (min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0))

    def mousePressEvent(self, e):
        if not self.brush_on or self._pix is None:
            return super().mousePressEvent(e)
        b = e.button()
        if b not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return
        left = (b == Qt.MouseButton.LeftButton) != self.swap_buttons
        self._mode = "glasses" if left else "original"
        q = self._to_img(e.position())
        self._pts = [q] if q else []
        self.update()

    def mouseMoveEvent(self, e):
        self._hover = e.position() if self.brush_on else None
        if self._mode is not None:
            q = self._to_img(e.position())
            if q and (not self._pts or q != self._pts[-1]):
                self._pts.append(q)
        if self.brush_on:
            self.update()
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._mode is not None:
            mode, pts = self._mode, self._pts
            self._mode, self._pts = None, []
            if pts:
                self.stroke_done.emit(mode, self.radius, pts)
            self.update()
            return
        super().mouseReleaseEvent(e)

    def leaveEvent(self, e):
        self._hover = None
        self.update()
        super().leaveEvent(e)

    def contextMenuEvent(self, e):
        if not self.brush_on:
            super().contextMenuEvent(e)

    def paintEvent(self, e):
        super().paintEvent(e)
        if not self.brush_on or self._pix is None:
            return
        r = self._target()
        rad = self.radius * r.width()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._mode is not None and self._pts:
            c = self.COLORS[self._mode]
            pts = [QPointF(r.left() + x * r.width(), r.top() + y * r.height()) for x, y in self._pts]
            if len(pts) == 1:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(c)
                p.drawEllipse(pts[0], rad, rad)
            else:
                p.setPen(QPen(c, 2 * rad, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
                for a, b in zip(pts, pts[1:]):
                    p.drawLine(a, b)
        if self._hover is not None:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(255, 255, 255), 1))
            p.drawEllipse(self._hover, rad, rad)
            p.setPen(QPen(QColor(0, 0, 0), 1, Qt.PenStyle.DotLine))
            p.drawEllipse(self._hover, rad, rad)


class FineSlider(QWidget):
    """0~200% 미세 슬라이더 (설계서 2-3, v1.52 조작 방식).

    더블클릭 = 기본값, 방향키 = 1%, Shift+방향키 = 5%, 숫자 직접 입력.
    valueChanged는 움직이는 동안, committed는 손을 뗐을 때(또는 숫자 입력·키 조작 후) 나온다.
    """
    valueChanged = Signal(int)
    committed = Signal(int)

    def __init__(self, label: str, default: int = 100, lo: int = 0, hi: int = 200, ticks=None, parent=None):
        super().__init__(parent)
        self.default = default
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.name = QLabel(label)
        self.name.setMinimumWidth(70)
        self.slider = _Slider(Qt.Orientation.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setValue(default)
        self.slider.ticks = ticks or [int(v * 100) for v in LEVELS.values()]
        self.spin = QSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setSuffix("%")
        self.spin.setValue(default)
        self.spin.setKeyboardTracking(False)
        lay.addWidget(self.name)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self.slider.valueChanged.connect(self._from_slider)
        self.slider.sliderReleased.connect(lambda: self.committed.emit(self.value()))
        self.slider.keyCommitted.connect(lambda: self.committed.emit(self.value()))
        self.slider.doubleClicked.connect(self.reset)
        self.spin.valueChanged.connect(self._from_spin)

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, v: int, emit: bool = False) -> None:
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(v)
        self.spin.setValue(v)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)
        if emit:
            self.valueChanged.emit(v)
            self.committed.emit(v)

    def reset(self) -> None:
        self.setValue(self.default, emit=True)

    def _from_slider(self, v: int) -> None:
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.valueChanged.emit(v)
        if not self.slider.isSliderDown() and not self.slider.keyboard_active:
            self.committed.emit(v)

    def _from_spin(self, v: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(v)
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)
        self.committed.emit(v)


class _Slider(QSlider):
    doubleClicked = Signal()
    keyCommitted = Signal()

    def __init__(self, *a):
        super().__init__(*a)
        self.ticks: list[int] = []
        self.keyboard_active = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def mouseDoubleClickEvent(self, e):
        self.doubleClicked.emit()

    def keyPressEvent(self, e):
        step = 5 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
        k = e.key()
        if k in (Qt.Key.Key_Left, Qt.Key.Key_Down):
            self.keyboard_active = True
            self.setValue(self.value() - step)
        elif k in (Qt.Key.Key_Right, Qt.Key.Key_Up):
            self.keyboard_active = True
            self.setValue(self.value() + step)
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if self.keyboard_active and not e.isAutoRepeat():
            self.keyboard_active = False
            self.keyCommitted.emit()
        super().keyReleaseEvent(e)

    def paintEvent(self, e):
        super().paintEvent(e)
        if not self.ticks:
            return
        p = QPainter(self)
        p.setPen(QPen(QColor(120, 120, 120), 1))
        span = self.maximum() - self.minimum()
        handle = 16
        for t in self.ticks:
            x = handle / 2 + (t - self.minimum()) / span * (self.width() - handle)
            p.drawLine(int(x), self.height() - 4, int(x), self.height())


class StrengthPanel(QWidget):
    """전문가 모드 부위별 강도 + [모두 약하게/기본/강하게]."""
    changed = Signal(bool)   # True = 손을 뗌(원본 해상도 계산), False = 움직이는 중(미리보기)

    def __init__(self, strength: Strength, parent=None):
        super().__init__(parent)
        self.strength = strength
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        for name in LEVELS:
            b = QPushButton(f"모두 {name}")
            b.clicked.connect(lambda _=False, n=name: self.set_all(n))
            row.addWidget(b)
        grid.addLayout(row, 0, 0)
        self.sliders: dict[str, FineSlider] = {}
        for i, (pid, label) in enumerate(PARTS.items(), start=1):
            s = FineSlider(label)
            s.setValue(round(strength.get(pid) * 100))
            s.valueChanged.connect(lambda v, p=pid: self._on(p, v, False))
            s.committed.connect(lambda v, p=pid: self._on(p, v, True))
            grid.addWidget(s, i, 0)
            self.sliders[pid] = s

    def bind(self, strength: Strength) -> None:
        self.strength = strength
        for pid, s in self.sliders.items():
            s.setValue(round(strength.get(pid) * 100))

    def set_all(self, level: str) -> None:
        self.strength.set_all(level)
        self.bind(self.strength)
        self.changed.emit(True)

    def _on(self, pid: str, v: int, final: bool) -> None:
        self.strength.set(pid, v / 100)
        self.changed.emit(final)
