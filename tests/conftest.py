"""Shared isolation for Qt GUI tests.

Many GUI modules intentionally reuse one QApplication.  A test that leaves a
hidden top-level widget behind must not make later theme or layout tests walk
an ever-growing widget tree.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QThread
from PySide6.QtWidgets import QApplication


@pytest.fixture(autouse=True)
def dispose_qt_top_level_widgets_after_test():
    yield
    app = QApplication.instance()
    if app is None:
        return
    # Delete abandoned test windows only when they do not own active work.
    # A running child QThread must remain under the application's normal close
    # guard; forcibly destroying its parent makes Qt abort the process.
    for widget in tuple(app.topLevelWidgets()):
        if any(thread.isRunning() for thread in widget.findChildren(QThread)):
            continue
        widget.hide()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
