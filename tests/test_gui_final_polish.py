from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractItemView, QApplication, QDialog, QFileDialog, QMessageBox, QWidget

import pyLOCO.gui.main_window as main_window_module
from pyLOCO.gui.main_window import ElementSelectionDialog, MainWindow
from pyLOCO.gui.models.project import ProjectMetadata, json_safe


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_corrector_steps_are_one_context_sensitive_section(app, tmp_path, monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: None)
    source = tmp_path / "steps.npz"
    np.savez(source, hor=np.full(3, 1e-4), ver=np.array([2e-4, 2.5e-4]))
    window = MainWindow()
    window.project.mode = "Advanced"
    window._apply_mode_visibility()

    window.params_cmstep_file.setText(str(source))
    window.cmstep_mode.setCurrentIndex(window.cmstep_mode.findData("file"))
    app.processEvents()
    assert not window.params_cmstep_file_row.isHidden()
    assert not window.params_cmstep_resolved.isHidden()
    assert window.params_cmstep_h.isHidden()
    assert "0.0001 rad from file" in window.params_cmstep_resolved_h.text()
    assert "2 values from file" in window.params_cmstep_resolved_v.text()

    window.cmstep_mode.setCurrentIndex(window.cmstep_mode.findData("uniform"))
    app.processEvents()
    assert not window.params_cmstep_h.isHidden()
    assert not window.params_cmstep_v.isHidden()
    assert window.params_cmstep_file_row.isHidden()
    assert window.params_cmstep_resolved.isHidden()
    # Legacy response-matrix state remains serializable, but it is not a
    # second user-facing pair of controls.
    assert window.rm_dkick_h.isHidden()
    assert window.rm_dkick_v.isHidden()
    window.project.modified = False
    window.close()


def test_new_project_returns_to_dashboard_without_legacy_individuals(app):
    window = MainWindow()
    window.project.mode = "Advanced"
    window._apply_mode_visibility()
    window._workspace.setCurrentWidget(window.results_page)
    window.project.modified = False

    window.new_project()
    app.processEvents()

    assert window._workspace.currentIndex() == 0
    assert window._workspace.tabText(0) == "Project"
    assert window.loco_individuals.isHidden()
    assert window.fit_workflow_table.rowCount() == 1
    window.project.modified = False
    window.close()


def test_fit_workflow_row_selection_loads_each_stage_configuration(app):
    window = MainWindow()
    assert window.fit_workflow_table.selectionBehavior() == QAbstractItemView.SelectRows
    assert window.fit_workflow_table.selectionMode() == QAbstractItemView.SingleSelection

    window.parameter_checks["hbpm_gain"].setChecked(True)
    window.parameter_checks["quads"].setChecked(False)
    window._store_active_fit_stage()
    window._duplicate_fit_stage()

    window.parameter_checks["hbpm_gain"].setChecked(False)
    window.parameter_checks["quads"].setChecked(True)
    window._store_active_fit_stage()

    window.fit_workflow_table.selectRow(0)
    app.processEvents()
    assert window.parameter_checks["hbpm_gain"].isChecked()
    assert not window.parameter_checks["quads"].isChecked()
    assert "Editing stage 1" in window.fit_stage_editor_label.text()

    window.fit_workflow_table.selectRow(1)
    app.processEvents()
    assert not window.parameter_checks["hbpm_gain"].isChecked()
    assert window.parameter_checks["quads"].isChecked()
    assert "Editing stage 2" in window.fit_stage_editor_label.text()
    window.project.modified = False
    window.close()


def test_backend_only_fit_recipe_stages_are_editable_in_gui(app):
    window = MainWindow()
    window.parameter_checks["hbpm_gain"].setChecked(True)
    window.parameter_checks["quads"].setChecked(False)
    window._store_active_fit_stage()
    window._duplicate_fit_stage()
    window.parameter_checks["hbpm_gain"].setChecked(False)
    window.parameter_checks["quads"].setChecked(True)
    window._store_active_fit_stage()

    for stage in window.fit_recipe.stages:
        stage.configuration.pop("gui_config", None)
    window._active_fit_stage = -1
    window.fit_workflow_table.clearSelection()
    window._refresh_fit_workflow_table(0)
    app.processEvents()
    assert window.parameter_checks["hbpm_gain"].isChecked()
    assert not window.parameter_checks["quads"].isChecked()

    window.fit_workflow_table.selectRow(1)
    app.processEvents()
    assert not window.parameter_checks["hbpm_gain"].isChecked()
    assert window.parameter_checks["quads"].isChecked()
    window.project.modified = False
    window.close()


def test_run_monitor_is_visible_when_run_preparation_fails(app, monkeypatch):
    messages = []
    monkeypatch.setattr(ProjectMetadata, "validation_messages", lambda _self: [])
    monkeypatch.setattr(
        main_window_module.LocoRunRequest,
        "from_project",
        lambda _project: (_ for _ in ()).throw(RuntimeError("preflight setup failed")),
    )
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, title, message: messages.append((title, message)),
    )
    window = MainWindow()

    window.run_loco()
    app.processEvents()

    assert window.results_workspace.run_status_label.text() == "Run preparation failed"
    assert window.results_workspace.run_iteration_label.text() == "Not started"
    assert "preflight setup failed" in window.results_workspace.log.text.toPlainText()
    assert messages and messages[0][0] == "Cannot start LOCO"
    assert window._run_thread is None
    window.project.modified = False
    window.close()


def test_run_switches_to_results_before_stage_snapshot(app, monkeypatch):
    window = MainWindow()
    observed = []

    def fail_snapshot():
        observed.append(
            (
                window._workspace.currentWidget() is window.results_page,
                window.results_workspace.run_status_label.text(),
            )
        )
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(window, "_store_active_fit_stage", fail_snapshot)
    monkeypatch.setattr(QMessageBox, "critical", lambda *_args, **_kwargs: None)

    window.run_loco()
    app.processEvents()

    assert observed == [(True, "Preparing inputs and checking the FIT workflow…")]
    assert window.results_workspace.run_status_label.text() == "Run preparation failed"
    window.project.modified = False
    window.close()


def test_advanced_element_previews_reuse_one_lattice_load(app, monkeypatch):
    window = MainWindow()
    window.project.mode = "Advanced"
    window.project.lattice.path = "/tmp/reference.mat"
    window._element_preview_signature = None
    calls = []
    monkeypatch.setattr(
        window,
        "_load_current_lattice",
        lambda: (calls.append(True) or []),
    )

    window._refresh_element_selection_ui()
    window._refresh_element_selection_ui()

    assert len(calls) == 1
    window.project.modified = False
    window.close()


def test_open_project_works_in_windowed_mode_with_qt_dialog(app, monkeypatch):
    calls = []

    def choose_project(*args, **kwargs):
        calls.append((args, kwargs))
        return "", ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", choose_project)
    window = MainWindow()
    window.resize(900, 650)
    window.show()
    app.processEvents()

    assert window.dashboard_open_project_button.isEnabled()
    QTest.mouseClick(window.dashboard_open_project_button, Qt.LeftButton)
    app.processEvents()

    assert len(calls) == 1
    assert calls[0][1]["options"] == QFileDialog.Option.DontUseNativeDialog
    window.open_project_action.trigger()
    assert len(calls) == 2
    window.project.modified = False
    window.close()


def test_open_project_does_not_prompt_for_unchanged_focused_dashboard_fields(app, monkeypatch):
    calls = []
    prompts = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (calls.append((args, kwargs)) or ("", "")),
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: (prompts.append((args, kwargs)) or QMessageBox.Cancel),
    )
    window = MainWindow()
    window.resize(1000, 700)
    window.show()
    window.project.modified = False
    window.dashboard_description.setFocus()
    app.processEvents()

    QTest.mouseClick(window.dashboard_open_project_button, Qt.LeftButton)
    app.processEvents()

    assert not prompts
    assert len(calls) == 1
    assert calls[0][1]["options"] == QFileDialog.Option.DontUseNativeDialog
    assert not window.project.modified
    window.close()


def test_numerical_jacobians_hide_analytical_options_and_restore_state(app, tmp_path):
    window = MainWindow()
    window.analytical_thick_quadrupole.setChecked(True)
    window.analytical_skew_verbose.setChecked(True)
    window.quad_jacobian_calculator.setCurrentText("Numerical")
    window.skew_jacobian_calculator.setCurrentText("Numerical")
    app.processEvents()
    assert window.normal_analytical_options.isHidden()
    assert window.skew_analytical_options.isHidden()

    window.quad_jacobian_calculator.setCurrentText("Analytical")
    window.skew_jacobian_calculator.setCurrentText("Analytical")
    app.processEvents()
    assert not window.normal_analytical_options.isHidden()
    assert not window.skew_analytical_options.isHidden()
    assert window.analytical_thick_quadrupole.isChecked()
    assert window.analytical_skew_verbose.isChecked()

    window.project.loco_config = window._collect_loco_configuration()
    path = window.project.save(tmp_path / "jacobian-options.pyloco.json")
    window.close()
    restored = MainWindow()
    restored.project = ProjectMetadata.load(path)
    restored._load_config_to_widgets()
    assert restored.quad_jacobian_calculator.currentText() == "Analytical"
    assert restored.skew_jacobian_calculator.currentText() == "Analytical"
    assert restored.analytical_thick_quadrupole.isChecked()
    assert restored.analytical_skew_verbose.isChecked()
    restored.close()


@pytest.mark.parametrize(
    "role_key,class_name",
    [
        ("bpm_ords", "Monitor"),
        ("horizontal_corrector_ords", "Corrector"),
        ("vertical_corrector_ords", "Corrector"),
        ("normal_quadrupole_ords", "Quadrupole"),
        ("skew_quadrupole_ords", "Quadrupole"),
        ("cavity_ords", "RFCavity"),
    ],
)
def test_manual_ordinals_populate_preview_for_every_component(app, role_key, class_name):
    element_type = type(class_name, (), {})
    lattice = [element_type(), element_type(), element_type()]
    for index, element in enumerate(lattice):
        element.FamName = f"E{index}"

    class Parent(QWidget):
        def _load_current_lattice(self):
            return lattice

    parent = Parent()
    dialog = ElementSelectionDialog(parent, role_key, [])
    dialog.manual_radio.setChecked(True)
    dialog.manual_edit.setPlainText("2, 0")
    assert dialog._preview()
    assert dialog.table.rowCount() == 2
    assert dialog.table.item(0, 0).text() == "0"
    assert dialog.table.item(0, 1).text() == "2"
    assert dialog.table.item(0, 2).text() == "E2"
    assert dialog.table.item(0, 3).text() == class_name
    assert dialog.table.item(1, 1).text() == "0"
    dialog.close(); parent.close()


def test_invalid_manual_ordinal_has_inline_validation(app, monkeypatch):
    monitor_type = type("Monitor", (), {})
    lattice = [monitor_type()]
    lattice[0].FamName = "BPM1"

    class Parent(QWidget):
        def _load_current_lattice(self):
            return lattice

    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: None)
    parent = Parent(); dialog = ElementSelectionDialog(parent, "bpm_ords", [])
    dialog.manual_edit.setPlainText("4")
    assert not dialog._preview()
    assert dialog.validation_message.isVisibleTo(dialog)
    assert "lattice range [0, 0]" in dialog.validation_message.text()
    assert dialog.table.rowCount() == 0
    dialog.close(); parent.close()


def test_recursive_json_safe_conversion_does_not_mutate_runtime_values(tmp_path):
    nested = {
        "array": [np.asarray([1.0e-4, 2.0e-4])],
        "integer": np.int64(7),
        "floating": np.float64(2.5),
        "boolean": np.bool_(True),
        "path": tmp_path / "steps.npz",
        "tuple": (np.int32(3), np.asarray([False, True], dtype=np.bool_)),
    }
    converted = json_safe(nested)
    assert converted == {
        "array": [[1.0e-4, 2.0e-4]], "integer": 7, "floating": 2.5,
        "boolean": True, "path": str(tmp_path / "steps.npz"),
        "tuple": [3, [False, True]],
    }
    json.dumps(converted)
    assert isinstance(nested["array"][0], np.ndarray)


def test_file_backed_dkick_survives_open_edit_save_and_reopen(
    app, tmp_path, monkeypatch
):
    steps = tmp_path / "corrector_steps.npz"
    horizontal = np.asarray([1.0e-4, 1.5e-4])
    vertical = np.asarray([2.0e-4])
    np.savez(steps, hor=horizontal, ver=vertical)

    project = ProjectMetadata(name="NumPy dkick regression")
    project.loco_config.machine_elements.bpm_ords = [4]
    project.loco_config.machine_elements.horizontal_corrector_ords = [10, 11]
    project.loco_config.machine_elements.vertical_corrector_ords = [20]
    project.loco_config.parameters.cmstep.mode = "file"
    project.loco_config.parameters.cmstep.file = str(steps)
    project_file = project.save(tmp_path / "portable.pyloco.json")
    assert str(tmp_path) not in project_file.read_text(encoding="utf-8")

    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: None)
    window = MainWindow()
    window.open_project(project_file)
    mapping = window.project.loco_config.to_backend_mapping()
    runtime_h, runtime_v = mapping["RMConfig"]["dkick"]
    assert isinstance(runtime_h, np.ndarray)
    assert isinstance(runtime_v, np.ndarray)
    assert runtime_h.shape == (2,) and runtime_v.shape == (1,)
    np.testing.assert_array_equal(runtime_h, horizontal)
    np.testing.assert_array_equal(runtime_v, vertical)
    window._update_fit_summary()
    assert '"dkick": [' in window.fit_summary.toPlainText()

    class AcceptedSelection:
        def __init__(self, _parent, _role_key, _current):
            self.selected_ords = [4, 5]

        def exec(self):
            return QDialog.Accepted

    monkeypatch.setattr(main_window_module, "ElementSelectionDialog", AcceptedSelection)
    window.edit_element_selection("bpm_ords")
    assert window.project.loco_config.machine_elements.bpm_ords == [4, 5]
    assert '"dkick": [' in window.fit_summary.toPlainText()

    window.save_project()
    saved_text = project_file.read_text(encoding="utf-8")
    assert str(tmp_path) not in saved_text
    window.project.modified = False
    window.close()

    restored = ProjectMetadata.load(project_file)
    restored_mapping = restored.loco_config.to_backend_mapping()
    restored_h, restored_v = restored_mapping["RMConfig"]["dkick"]
    assert isinstance(restored_h, np.ndarray) and isinstance(restored_v, np.ndarray)
    np.testing.assert_array_equal(restored_h, horizontal)
    np.testing.assert_array_equal(restored_v, vertical)
    assert Path(restored.loco_config.parameters.cmstep.file) == steps
