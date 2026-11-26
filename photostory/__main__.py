"""실행: python -m photostory"""
from __future__ import annotations

import logging
import sys


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from PySide6.QtWidgets import QApplication

    from .app.main_window import MainWindow
    from .app.style import STYLE

    app = QApplication(sys.argv)
    app.setApplicationName("PhotoStory")
    app.setStyleSheet(STYLE)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
