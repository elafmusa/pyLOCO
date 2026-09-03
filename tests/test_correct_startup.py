"""Correct startup and shared-shell regressions; never contact a machine."""
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QSettings
from PySide6.QtWidgets import QApplication
from pyLOCO.gui import appearance
from pyLOCO.gui.themes import apply_application_theme, theme_for_key
from pyLOCO.correct.app import build_application
from pyLOCO.correct.main_window import CorrectMainWindow, AMBER_QSS


@pytest.fixture
def settings(tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "suite.ini"), QSettings.IniFormat)
    monkeypatch.setattr(appearance, "suite_appearance_settings", lambda: settings)
    return settings


@pytest.mark.parametrize("key", ["dark", "light"])
def test_empty_correct_inherits_without_restyling_or_backend_initialization(key, settings, monkeypatch):
    app = QApplication.instance() or build_application(["correct-startup-test"])
    apply_application_theme(app, theme_for_key(key))
    original = app.styleSheet()
    def forbidden(*args, **kwargs):
        raise AssertionError("startup must not create canvases or connect to a machine")
    monkeypatch.setattr("pyLOCO.correct.main_window.PlotCanvas", forbidden)
    window = CorrectMainWindow()
    assert window.theme_key == key and app.styleSheet() == original
    assert window.styleSheet() == AMBER_QSS
    assert window.review is None and window.backend_session is None and window.machine_snapshot is None
    assert not window.plots and window.table.rowCount() == 0
    assert window.mapping_path is None and not window.apply_button.isEnabled()
    assert not window.windowIcon().isNull()
    window.close(); window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.mark.parametrize("key", ["dark", "light"])
def test_actual_fit_handler_theme_toggle_and_reopen(key, settings, monkeypatch):
    from pyLOCO.gui import main_window as fit
    monkeypatch.setattr(fit, "QSettings", lambda: settings)
    settings.setValue("appearance/theme", key)
    app = QApplication.instance() or build_application(["suite-theme-test"])
    owner = fit.MainWindow(); owner.show(); app.processEvents()
    original = app.styleSheet()
    assert owner.open_correct_app()
    correct = owner._correct_window
    assert correct.theme_key == key and app.styleSheet() == original
    correct.toggle_theme(); app.processEvents()
    changed = "light" if key == "dark" else "dark"
    assert owner.current_theme.key == correct.theme_key == changed
    assert settings.value("appearance/theme") == changed
    correct.close(); QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert owner.open_correct_app()
    assert owner._correct_window.theme_key == changed
    # Suite-originated changes also refresh Correct's own theme button/state.
    owner._apply_theme_selection(key); app.processEvents()
    assert owner._correct_window.theme_key == key
    owner._correct_window.close(); owner.close(); owner.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


@pytest.mark.parametrize("key", ["dark", "light"])
def test_fresh_direct_correct_uses_persisted_suite_preference_without_heavy_imports(key, tmp_path):
    script = '''
import sys
from PySide6.QtCore import QSettings
from pyLOCO.gui import appearance
settings = QSettings(sys.argv[1], QSettings.IniFormat)
appearance.suite_appearance_settings = lambda: settings
from pyLOCO.correct.app import build_application, CorrectMainWindow
app = build_application(["pyloco-correct"])
window = CorrectMainWindow()
assert window.theme_key == sys.argv[2]
assert not any(name in sys.modules for name in ("at", "scipy", "matplotlib", "pyLOCO.pyloco"))
assert not window.plots and window.review is None and window.backend_session is None
window.toggle_theme()
assert settings.value("appearance/theme") != sys.argv[2]
window.close()
'''
    path = tmp_path / "shared.ini"
    settings = QSettings(str(path), QSettings.IniFormat)
    settings.setValue("appearance/theme", key); settings.sync()
    subprocess.run([sys.executable, "-c", script, str(path), key], check=True,
                   cwd=Path(__file__).resolve().parents[1], env=os.environ.copy())
    # A second fresh process must see the value persisted by the first toggle.
    changed = "light" if key == "dark" else "dark"
    subprocess.run([sys.executable, "-c", script, str(path), changed], check=True,
                   cwd=Path(__file__).resolve().parents[1], env=os.environ.copy())
