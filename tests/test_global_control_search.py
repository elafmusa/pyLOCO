from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QMessageBox, QToolBar

from pyLOCO.correct.main_window import CorrectMainWindow
from pyLOCO.gui.main_window import MainWindow
from pyLOCO.measure.main_window import MeasureMainWindow
from pyLOCO.gui.models.project import ProjectMetadata


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _result(controller, label):
    return next(result for result in controller.search(label) if result.label == label)


def test_fit_uses_one_always_advanced_interface(app, tmp_path):
    legacy = ProjectMetadata(mode="Basic")
    source = legacy.save(tmp_path / "legacy-basic.pyloco.json")
    window = MainWindow()
    window.project.modified = False
    window.open_project(source)
    app.processEvents()
    toolbar = window.findChild(QToolBar, "mainToolbar")
    action_labels = {_action.text().replace("&", "") for _action in toolbar.actions()}
    assert "Basic" not in action_labels
    assert "Advanced" not in action_labels
    assert window.project.mode == "Advanced"
    window._apply_mode_visibility()
    assert not window.rm_fixedpath.isHidden()
    assert not window.rm_coupling.isHidden()
    assert not window.fixed_group.isHidden()
    window.project.modified = False
    window.close()


def test_fit_toolbar_reflows_secondary_controls_without_hiding_primary_actions(app):
    window = MainWindow(); window.show(); app.processEvents(); window.resize(1600, 900); app.processEvents()
    toolbar = window.findChild(QToolBar, "mainToolbar")
    assert not window.more_toolbar_action.isVisible()
    assert window.appearance_toolbar_action.isVisible()
    assert window.screenshot_toolbar_action.isVisible()
    assert window.search_button_action.isVisible()
    assert window.search_box_action is None
    assert not window.control_search.search_editor.isVisible()
    assert window.screenshot_controller.button.text() == "📷"
    assert window.screenshot_controller.button.accessibleName() == "Save Screenshot"
    assert window.appearance_button.accessibleName() == "Appearance"
    assert window.control_search.search_button.accessibleName() == "Find Setting or Action"

    window.resize(900, 700); app.processEvents()
    assert window.more_toolbar_action.isVisible()
    assert not window.appearance_toolbar_action.isVisible()
    assert not window.screenshot_toolbar_action.isVisible()
    assert not window.search_button_action.isVisible()
    for action in (
        window.new_project_action, window.open_project_action,
        window.save_project_action, window.validate_project_action,
        window.run_loco_action,
    ):
        button = toolbar.widgetForAction(action)
        assert action.isVisible() and button is not None and button.isVisible()
        assert button.geometry().right() <= toolbar.width()
    assert window.continue_loco_action in toolbar.actions()
    tabs = window._workspace.tabBar()
    assert tabs.tabRect(tabs.count() - 1).right() <= tabs.width()
    window.project.modified = False; window.close()


def test_fit_compact_search_and_more_menu_preserve_existing_actions(app):
    window = MainWindow(); window.show(); app.processEvents(); window.resize(900, 700); app.processEvents()
    labels = [action.text().replace("&", "") for action in window.more_menu.actions()]
    assert "📷 Save Screenshot" in labels
    assert "Save Screenshot As…" in labels
    assert "Appearance" in labels
    assert "Find Setting or Action…" in labels
    window.find_control_action.trigger()
    app.processEvents()
    assert window.control_search.search_box.isVisible()
    assert window.control_search.search_box.hasFocus()
    window.project.modified = False; window.close()


def test_fit_search_finds_documented_examples_and_navigates_without_activation(app):
    window = MainWindow()
    window.show()
    app.processEvents()
    search = window.control_search

    assert any(result.label == "Save Jacobian" for result in search.search("Jacobian"))
    assert any("Normal Quadrupoles" in result.label for result in search.search("Quadrupole"))
    assert any(result.label == "Save Screenshot" for result in search.search("Screenshot"))
    assert any(result.label == "Find Machine Mapping" for result in search.search("Mapping"))

    fixed_path = _result(search, "fixedpathlength")
    assert fixed_path.widget is window.rm_fixedpath
    assert not fixed_path.requires_advanced
    original = window.rm_fixedpath.isChecked()
    assert search.navigate_to(fixed_path)
    app.processEvents()
    assert window.project.mode == "Advanced"
    assert window._workspace.tabText(window._workspace.currentIndex()) == "Fit"
    assert window.rm_fixedpath.isChecked() is original

    search.shortcut.activated.emit()
    assert search.search_box.hasFocus()
    window.project.modified = False
    window.close()


def test_search_result_rows_show_control_and_location(app):
    window = MainWindow()
    window.show()
    window.control_search.search_box.setText("Mapping")
    app.processEvents()
    assert window.control_search.result_list.count() > 0
    first = window.control_search.result_list.item(0).text()
    assert "—" in first
    assert "Machine Components" in "\n".join(
        window.control_search.result_list.item(row).text()
        for row in range(window.control_search.result_list.count())
    )
    window.project.modified = False
    window.close()


def test_measure_search_explains_disabled_control_without_clicking(app, monkeypatch):
    window = MeasureMainWindow()
    window.show()
    app.processEvents()
    search = window.control_search
    assert any(result.label == "Find Machine Mapping" for result in search.search("Mapping"))
    assert any(result.label == "Save Screenshot" for result in search.search("Screenshot"))

    target = _result(search, "Open in FIT")
    spy = QSignalSpy(window.session_toolbar_buttons["Open in FIT"].clicked)
    messages = []
    monkeypatch.setattr(
        QMessageBox, "information",
        lambda _parent, title, message, *_args, **_kwargs: messages.append((title, message)),
    )
    assert not target.widget.isEnabled()
    assert search.navigate_to(target)
    app.processEvents()
    assert spy.count() == 0
    assert not target.widget.isEnabled()
    assert messages and "disabled" in messages[-1][1]
    window._session_dirty = False
    window.close()


def test_correct_search_opens_mapping_tab_without_activating_action(app, monkeypatch):
    window = CorrectMainWindow()
    window.show()
    app.processEvents()
    search = window.control_search
    mapping = _result(search, "Load Machine Mapping")
    spy = QSignalSpy(mapping.widget.clicked)
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: None)
    assert search.navigate_to(mapping)
    app.processEvents()
    assert window.tabs.currentIndex() == 1
    assert spy.count() == 0
    assert any(result.label == "Save Screenshot" for result in search.search("Screenshot"))
    window.close()
