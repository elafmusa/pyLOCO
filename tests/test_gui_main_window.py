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
