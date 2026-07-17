"""Regression tests for GUI main-window refresh wiring."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


def test_fit_configuration_refresh_has_run_loco_action() -> None:
    """Changing fit controls repeatedly should not reference a missing action."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from pyLOCO.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    assert app is not None
    window = MainWindow()
    try:
        assert hasattr(window, "run_loco_action")

        window._on_fit_config_changed()
        window._on_fit_config_changed()

        assert window.run_loco_action.isEnabled() == window.project.is_complete
    finally:
        window.close()


def test_refresh_ui_qactions_are_created() -> None:
    """Every *_action used by _refresh_ui should be assigned by _create_actions."""
    import ast
    from pathlib import Path

    source = Path("pyLOCO/gui/main_window.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in {"_refresh_ui", "_create_actions"}
    }

    refresh_actions = {
        node.attr
        for node in ast.walk(methods["_refresh_ui"])
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr.endswith("_action")
    }
    created_actions = {
        target.attr
        for node in ast.walk(methods["_create_actions"])
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
        and target.attr.endswith("_action")
    }

    assert refresh_actions <= created_actions




def test_gui_debug_argument_is_removed_before_qt() -> None:
    """The GUI debug flag should enable diagnostics without being passed to Qt."""
    import sys
    from pathlib import Path

    pytest.importorskip("PySide6")
    sys.path.insert(0, str(Path.cwd()))
    from pyLOCO.gui.app import DEBUG_GUI_FLAG, _debug_requested, _qt_argv

    argv = ["pyloco-gui", DEBUG_GUI_FLAG, "--platform", "offscreen"]

    assert _debug_requested(argv) is True
    assert _qt_argv(argv) == ["pyloco-gui", "--platform", "offscreen"]

def test_theme_controls_are_exposed_in_menus_and_toolbar() -> None:
    """Theme switching should be discoverable from View, Settings, and the toolbar."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from pyLOCO.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    assert app is not None
    window = MainWindow()
    try:
        menu_titles = [action.text().replace("&", "") for action in window.menuBar().actions()]
        assert "View" in menu_titles
        assert "Settings" in menu_titles

        assert window.menuBar().isNativeMenuBar() is False
        assert hasattr(window, "theme_actions")
        assert {action.text() for action in window.theme_actions.values()} == {"Light", "Dark"}
        assert window.toggle_theme_action.text().startswith(("☀️", "🌙"))
        assert window.appearance_action.text() == "Appearance…"
    finally:
        window.close()

def test_gui_backend_uses_internal_config_without_legacy_module() -> None:
    """GUI backend configuration should not require an external pyloco_config.py."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path.cwd()))
    from pyLOCO.gui.backend import _make_gui_config
    from pyLOCO.gui.models.project import LocoConfiguration

    sys.modules.pop("pyloco_config", None)
    mapping = LocoConfiguration().to_backend_mapping()

    config_module = _make_gui_config(mapping)

    assert config_module.__name__ == "pyLOCO.config"
    assert "pyloco_config" not in sys.modules
    assert config_module.RMConfig(**mapping["RMConfig"]).dkick == (1e-5, 1e-5)
    assert config_module.FitInitConfig(**mapping["FitInitConfig"]).fit_list == mapping["FitInitConfig"]["fit_list"]


def test_response_matrix_config_preserves_backend_ranges_and_legacy_calculators() -> None:
    """GUI RM configuration should match backend defaults and normalize equivalent algorithms."""
    from pyLOCO.gui.models.project import LocoConfiguration, ResponseMatrixConfig

    cfg = ResponseMatrixConfig(calculator="PyAT", rfStep=-1234.5, dkick_h=1e-6, dkick_v=2.5e-4)
    kwargs = cfg.to_rm_config_kwargs()

    assert kwargs["calculator"] == "Tracking"
    assert kwargs["rfStep"] == -1234.5
    assert kwargs["dkick"] == (1e-6, 2.5e-4)

    defaults = LocoConfiguration().to_backend_mapping()["RMConfig"]
    assert defaults["dkick"] == (1e-5, 1e-5)
    assert defaults["rfStep"] == -3000.0


def test_bad_bpm_validation_and_preprocessing() -> None:
    np = pytest.importorskip("numpy")

    from pyLOCO.gui.backend import _apply_bad_bpm_positions, _as_bad_bpm_positions
    from pyLOCO.pyloco import remove_bad_bpms

    positions = _as_bad_bpm_positions(np.array([1, 3]))
    measured = {
        "noise_x": np.arange(5),
        "noise_y": np.arange(10, 15),
        "eta_x": np.arange(20, 25),
        "eta_y": np.arange(30, 35),
        "orm": np.arange(10 * 4).reshape(10, 4),
    }
    indices = {"used_bpms_ords": np.arange(100, 105), "nHBPM": 5, "nVBPM": 5}

    cleaned, cleaned_indices = _apply_bad_bpm_positions(measured, indices, positions, remove_bad_bpms)

    assert cleaned_indices["used_bpms_ords"].tolist() == [100, 102, 104]
    assert cleaned_indices["nHBPM"] == 3
    assert cleaned_indices["nVBPM"] == 3
    assert cleaned["noise_x"].tolist() == [0, 2, 4]
    assert cleaned["noise_y"].tolist() == [10, 12, 14]
    assert cleaned["eta_x"].tolist() == [20, 22, 24]
    assert cleaned["eta_y"].tolist() == [30, 32, 34]
    assert cleaned["orm"].shape == (6, 4)

    with pytest.raises(ValueError, match="one-dimensional"):
        _as_bad_bpm_positions(np.ones((2, 2), dtype=int))
    with pytest.raises(ValueError, match="integer"):
        _as_bad_bpm_positions(np.array([1.2]))
    with pytest.raises(ValueError, match="unique"):
        _as_bad_bpm_positions(np.array([1, 1]))
    with pytest.raises(ValueError, match="valid 0-based BPM position range"):
        _apply_bad_bpm_positions(measured, indices, np.array([5]), remove_bad_bpms)


def test_orm_comparison_window_decimates_for_rendering() -> None:
    """The ORM viewer should keep full arrays while rendering a smaller mesh."""
    pytest.importorskip("PySide6")
    np = pytest.importorskip("numpy")
    pytest.importorskip("matplotlib")
    from PySide6.QtWidgets import QApplication

    from pyLOCO.gui.widgets.orm_comparison import OrmComparisonWindow

    app = QApplication.instance() or QApplication([])
    assert app is not None
    measured = np.arange(300 * 500, dtype=float).reshape(300, 500)
    model = measured + 1.0
    window = OrmComparisonWindow(measured, model)
    try:
        rendered = window._decimate(measured, max_points=10_000)
        assert window.measured_orm.shape == (300, 500)
        assert window.difference_orm.shape == (300, 500)
        assert rendered.values.size <= 10_000
        assert "RMS" in window.rms_label.text()
    finally:
        window.close()
