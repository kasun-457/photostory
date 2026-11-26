"""화면 흐름 연기 시험 (화면 없이 offscreen으로)."""
import os
import time

import cv2
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtCore import Qt  # noqa: E402


def _wait(app, cond, timeout=60):
    t = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.01)
        if time.time() - t > timeout:
            raise TimeoutError
    app.processEvents()


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def test_full_flow(app, hub, portrait, tmp_path, monkeypatch):
    from photostory.app.main_window import MainWindow
    from photostory.app.settings import AppSettings
    src = tmp_path / "김철수_원본.png"
    cv2.imencode(".png", cv2.cvtColor(portrait, cv2.COLOR_RGB2BGR))[1].tofile(str(src))

    s = AppSettings.load()
    s.expert = True
    (tmp_path / "출력").mkdir()
    s.output_dir = str(tmp_path / "출력")
    s.fixed_output = True
    w = MainWindow(hub, s)
    w.show()
    cv = w.customer_view
    cv.name.setText("김철수")
    for i in range(cv.purposes.count()):
        it = cv.purposes.item(i)
        if it.data(256) in ("passport", "half"):
            it.setCheckState(Qt.CheckState.Checked)
    cv.next_btn.click()
    assert w.stack.currentWidget() is w.upload_view
    assert not w.upload_view.start_btn.isEnabled()
    w.upload_view.accept_files([src], "original")
    _wait(app, lambda: "얼굴 확인됨" in w.upload_view.original.status.text())
    w.upload_view.start_btn.click()
    _wait(app, lambda: w._full_result is not None)
    rv = w.result_view
    assert "a4" in w._full_result[1].frame.outputs
    rv.guides_cb.setChecked(True)
    app.processEvents()
    os.makedirs(tmp_path / "shots", exist_ok=True)
    w.grab().save(str(os.environ.get("SHOT_DIR", tmp_path / "shots")) + "/result.png")
    # 전문가 모드: 배치 미세 조정 → 다시 계산
    rv.placement.size.setValue(5, emit=True)
    _wait(app, lambda: w._full_result is not None and not w._preview_busy)
    # 출력 (고정 폴더)
    w.export()
    _wait(app, lambda: (tmp_path / "출력" / "김철수_a4출력.jpg").exists() or False, 30)
    files = sorted(p.name for p in (tmp_path / "출력").iterdir())
    assert files == ["김철수_a4출력.jpg", "김철수_수정본.jpg", "김철수_전송_반명함판.jpg", "김철수_전송_여권.jpg"]
    w.next_customer()
    assert w.stack.currentWidget() is w.customer_view and w.proc.pipeline.analysis is None
    w.runner.wait(5000)


def test_result_tools(app):
    """결과 화면 손질 도구: 안경 복원 붓 획이 세션에 쌓이고, 쌍꺼풀 기준 눈을 바꾸면 다시 계산한다."""
    import numpy as np
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    from photostory.app.result_view import ResultView
    from photostory.core.session import Customer, Session
    s = Session(Customer(name="a", purposes=["passport"], glasses=True))
    rv = ResultView()
    rv.bind(s, [], expert=False)
    rv.resize(900, 700)
    rv.show()
    rv.ret_view.set_image(np.full((400, 300, 3), 200, np.uint8))
    app.processEvents()
    hits = []
    rv.params_changed.connect(hits.append)
    assert rv.brush_cb.isVisibleTo(rv) and not rv.crease_box.isVisibleTo(rv)
    rv.brush_cb.setChecked(True)
    v = rv.ret_view
    c = v.rect().center()
    QTest.mousePress(v, Qt.MouseButton.RightButton, pos=c)
    QTest.mouseMove(v, c + QPoint(20, 5))
    QTest.mouseRelease(v, Qt.MouseButton.RightButton, pos=c + QPoint(20, 5))
    QTest.mouseClick(v, Qt.MouseButton.LeftButton, pos=c + QPoint(-30, 0))
    assert [st.mode for st in s.glasses_strokes] == ["original", "glasses"]
    assert all(0 <= x <= 1 and 0 <= y <= 1 for st in s.glasses_strokes for x, y in st.pts)
    rv.undo_btn.click()
    assert len(s.glasses_strokes) == 1
    rv.clear_btn.click()
    assert not s.glasses_strokes and not rv.clear_btn.isEnabled()
    rv._update_tools(crease={"왼눈": {"height": 30}, "오른눈": {"height": 40}})
    assert rv.crease_box.isVisibleTo(rv)
    rv.crease_box.setCurrentIndex(rv.crease_box.findData("오른눈"))
    assert s.crease_ref == "오른눈" and hits and all(hits)
