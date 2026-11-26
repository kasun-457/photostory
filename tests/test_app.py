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
