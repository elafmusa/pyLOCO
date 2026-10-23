from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QScrollArea

from pyLOCO.correct.main_window import CorrectMainWindow
from pyLOCO.gui.main_window import MainWindow
from pyLOCO.measure.main_window import MeasureMainWindow


def _app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize(
    "factory,size,slug",
    [
        (MeasureMainWindow, (1000, 700), "measure"),
        (MeasureMainWindow, (1280, 800), "measure"),
        (MainWindow, (900, 650), "fit"),
        (MainWindow, (1280, 800), "fit"),
        (CorrectMainWindow, (1000, 700), "correct"),
        (CorrectMainWindow, (1280, 800), "correct"),
    ],
)
def test_each_application_captures_visible_window_at_current_scroll(
    tmp_path, factory, size, slug
):
    app = _app()
    window = factory()
    window.resize(*size)
    window.show()
    app.processEvents()
    scrolls = window.findChildren(QScrollArea)
    if scrolls:
        bar = scrolls[0].verticalScrollBar()
        bar.setValue(min(bar.maximum(), max(1, bar.maximum() // 2)))
        scroll_position = bar.value()
    else:
        bar = None
        scroll_position = None
    target = tmp_path / slug / f"{slug}.png"
    saved = window.screenshot_controller.capture_to(target)
    assert saved == target.resolve()
    assert target.is_file()
    image = QImage(str(target))
    assert not image.isNull()
    assert image.width() == window.grab().width()
    assert image.height() == window.grab().height()
    if bar is not None:
        assert bar.value() == scroll_position
    assert str(target.resolve()) in window.statusBar().currentMessage()
    window.close()


def test_default_folder_uses_saved_session_or_project_and_fallback(tmp_path, monkeypatch):
    app = _app()
    measure = MeasureMainWindow()
    measure.saved_session_path = tmp_path / "measurements" / "session.pyloco-session.json"
    assert measure.screenshot_controller.default_directory() == tmp_path / "measurements" / "screenshots"

    fit = MainWindow()
    fit.project.path = str(tmp_path / "fit" / "project.pyloco.json")
    assert fit.screenshot_controller.default_directory() == tmp_path / "fit" / "screenshots"

    correct = CorrectMainWindow()
    correct.source_path.setText(str(tmp_path / "results"))
    (tmp_path / "results").mkdir()
    assert correct.screenshot_controller.default_directory() == tmp_path / "results" / "screenshots"

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    empty = MeasureMainWindow()
    assert empty.screenshot_controller.default_directory() == tmp_path / "Pictures" / "pyLOCO" / "Screenshots"
    for window in (measure, fit, correct, empty):
        window.close()
    app.processEvents()


def test_save_as_remembers_manual_directory_and_captures_after_dialog(tmp_path, monkeypatch):
    app = _app()
    window = MeasureMainWindow()
    remembered = {}
    window.screenshot_controller.settings = type(
        "MemorySettings", (), {
            "value": lambda self, key, default="": remembered.get(key, default),
            "setValue": lambda self, key, value: remembered.__setitem__(key, value),
        }
    )()
    window.show()
    app.processEvents()
    target = tmp_path / "chosen" / "operator-view"
    monkeypatch.setattr(
        "pyLOCO.gui.screenshots.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(target), "PNG image (*.png)"),
    )
    window.screenshot_controller.save_screenshot_as()
    app.processEvents()
    expected = target.with_suffix(".png")
    assert expected.is_file()
    assert remembered["screenshots/last_manual_directory"] == str(target.parent)
    assert window.findChild(type(window.screenshot_controller.button), "screenshotButton") is not None
    window.close()


def test_one_click_creates_timestamped_session_screenshot_folder(tmp_path):
    app = _app()
    window = MeasureMainWindow()
    window.saved_session_path = tmp_path / "session" / "measurement-session.pyloco-session.json"
    window.show()
    app.processEvents()
    saved = window.screenshot_controller.save_screenshot()
    assert saved is not None
    assert saved.parent == tmp_path / "session" / "screenshots"
    assert saved.name.startswith("pyloco-measure-")
    assert saved.suffix == ".png"
    assert saved.is_file()
    window.close()
