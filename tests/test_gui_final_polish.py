from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QLabel, QMessageBox, QPlainTextEdit, QWidget,
)

import pyLOCO.gui.main_window as main_window_module
from pyLOCO.gui.main_window import ElementSelectionDialog, MainWindow
from pyLOCO.gui.fit_workflow import FitRecipe
from pyLOCO.gui.models.project import (
    CompletedRunReference, ImportedDataset, LocoConfiguration, ProjectMetadata, json_safe,
)


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


def test_removing_stages_loads_the_surviving_stage_configuration(app):
    window = MainWindow()
    parameter_pairs = [
        ("hbpm_gain", "quads"),
        ("vbpm_gain", "quads"),
        ("quads", "skew_quads"),
        ("skew_quads", "hbpm_coupling"),
    ]

    # Build four stages with deliberately distinct configurations.
    first_on, first_off = parameter_pairs[0]
    window.parameter_checks[first_on].setChecked(True)
    window.parameter_checks[first_off].setChecked(False)
    window._store_active_fit_stage()
    for on_name, off_name in parameter_pairs[1:]:
        window._duplicate_fit_stage()
        for checkbox in window.parameter_checks.values():
            checkbox.setChecked(False)
        window.parameter_checks[on_name].setChecked(True)
        window.parameter_checks[off_name].setChecked(False)
        window._store_active_fit_stage()

    # Reopen stage 1, then remove the first three stages. Each deletion must
    # load the configuration of the stage that moved into row 1.
    window.fit_workflow_table.selectRow(0)
    app.processEvents()
    for expected_on, _expected_off in parameter_pairs[1:]:
        window._remove_fit_stage()
        app.processEvents()
        assert window.parameter_checks[expected_on].isChecked()
        assert window.project.modified is True
        assert "Unsaved changes" in window._project_label.text()
        assert "Editing stage 1" in window.fit_stage_editor_label.text()
        assert window.fit_stage_editor_label.text().endswith(
            "the FIT controls below belong to this stage"
        )

    assert len(window.fit_recipe.stages) == 1
    assert window.fit_recipe.stages[0].name == "Stage 1 copy copy copy"
    window.project.modified = False
    window.close()


def test_clear_fit_configuration_supports_empty_save_reopen_and_clean_rebuild(app, tmp_path, monkeypatch):
    lattice = tmp_path / "ring.mat"; lattice.write_bytes(b"lattice")
    orm = tmp_path / "orm.h5"
    with h5py.File(orm, "w") as handle:
        handle.create_dataset("response_matrix", data=np.zeros((4, 2)))
    original = tmp_path / "historical.pyloco.json"
    adapted = tmp_path / "adapted-empty.pyloco.json"

    window = MainWindow()
    window.project.lattice.path = str(lattice)
    window.project.lattice.file_type = "mat"
    window.project.lattice.mat_variable = "betamodel"
    window.project.measurements["orm"] = ImportedDataset("orm", str(orm), "h5", orm.stat().st_size)
    elements = window.project.loco_config.machine_elements
    elements.bpm_ords = [1, 2]
    elements.horizontal_corrector_ords = [3]
    elements.vertical_corrector_ords = [4]
    elements.normal_quadrupole_ords = [5]
    window.project.loco_config.element_selection_state = {
        "bpm_ords": {"method": "validated_measurement_mapping", "resolved_ords": [1, 2]}
    }
    window.project.measurement_device_mapping = {
        "schema": "pyloco-fit-measurement-mapping/v1", "mappings": [{"status": "mapped"}]
    }
    window.project.loco_config.bad_bpm_positions = [1]
    window.project.loco_config.excluded_horizontal_corrector_positions = [0]
    window.project.completed_run = CompletedRunReference(str(tmp_path / "old-results"), 1.0, "completed")
    window.project.fit_run_session = {"path": str(tmp_path / "old-session.json"), "completed_stages": 1}

    # The historical stage deliberately differs from initial defaults.
    window.solver_n_iter.setValue(2)
    for checkbox in window.parameter_checks.values():
        checkbox.setChecked(False)
    window.parameter_checks["skew_quads"].setChecked(True)
    window.parameter_checks["hbpm_coupling"].setChecked(True)
    window.parameter_checks["vbpm_coupling"].setChecked(True)
    window._store_active_fit_stage()
    window._snapshot_project_from_widgets()
    window.project.save(original)
    original_bytes = original.read_bytes()
    measurement_bytes = orm.read_bytes()

    monkeypatch.setattr(window, "_confirm_clear_fit_configuration", lambda exclusions: True)
    assert window.clear_fit_configuration() is True
    assert window.fit_recipe.stages == []
    assert window._active_fit_stage == -1
    assert not window.fit_empty_state_label.isHidden()
    assert window.fit_stage_buttons["+ Add stage"].text() == "+ Add first FIT stage"
    assert window.project.lattice.path == str(lattice)
    assert window.project.lattice.mat_variable == "betamodel"
    assert window.project.measurements["orm"].path == str(orm)
    assert window.project.loco_config.machine_elements.bpm_ords == [1, 2]
    assert window.project.loco_config.machine_elements.normal_quadrupole_ords == [5]
    assert window.project.loco_config.element_selection_state["bpm_ords"]["method"] == "validated_measurement_mapping"
    assert window.project.measurement_device_mapping["mappings"][0]["status"] == "mapped"
    assert window.project.loco_config.bad_bpm_positions == []
    assert window.project.loco_config.excluded_horizontal_corrector_positions == []
    assert window.project.completed_run.results_dir == ""
    assert window.project.fit_run_session == {}
    assert not window.run_loco_action.isEnabled()
    assert not window.resume_fit_run_button.isEnabled()
    assert not window.save_fit_recipe_button.isEnabled()
    assert not window.save_fit_run_session_button.isEnabled()
    assert original.read_bytes() == original_bytes
    assert orm.read_bytes() == measurement_bytes

    assert window._save_project_to(adapted) is True
    restored = ProjectMetadata.load(adapted)
    assert restored.fit_recipe["stages"] == []
    reopened = MainWindow()
    reopened.project = restored
    reopened._load_config_to_widgets()
    reopened._load_project_fit_recipe()
    reopened._refresh_ui("reopened")
    assert reopened.fit_recipe.stages == []
    assert not reopened.fit_empty_state_label.isHidden()
    assert not reopened.run_loco_action.isEnabled()

    # Adding and rebuilding starts from defaults, not the deleted skew stage.
    reopened._add_fit_stage()
    assert len(reopened.fit_recipe.stages) == 1
    assert reopened.fit_recipe.stages[0].start_from == "original_model"
    config = reopened._configuration_for_fit_stage(reopened.fit_recipe.stages[0])
    assert config.solver.nIter == 1
    assert config.parameters.skew_quads is False
    assert config.parameters.hbpm_coupling is False
    assert config.parameters.vbpm_coupling is False
    for _ in range(3):
        reopened._duplicate_fit_stage()
    assert len(reopened.fit_recipe.stages) == 4
    for stage in reopened.fit_recipe.stages:
        stage_config = reopened._configuration_for_fit_stage(stage)
        assert stage_config.solver.nIter == 1
        assert stage_config.parameters.skew_quads is False
        assert stage_config.parameters.hbpm_coupling is False
        assert stage_config.parameters.vbpm_coupling is False
    assert not reopened.run_loco_action.isEnabled()  # Full validation is still required.
    window.project.modified = False; window.close()
    reopened.project.modified = False; reopened.close()


def test_removing_final_fit_stage_enters_confirmed_empty_state(app, monkeypatch):
    window = MainWindow()
    assert len(window.fit_recipe.stages) == 1
    monkeypatch.setattr(window, "_confirm_clear_fit_configuration", lambda exclusions: True)
    window._remove_fit_stage()
    assert window.fit_recipe.stages == []
    assert window.fit_workflow_table.rowCount() == 0
    assert window._active_fit_stage == -1
    assert not window.run_loco_action.isEnabled()
    window.project.modified = False; window.close()


def test_clear_fit_configuration_cancel_is_transactional(app, monkeypatch):
    window = MainWindow()
    window.solver_n_iter.setValue(7)
    window._store_active_fit_stage()
    before_project = window.project.to_dict()
    before_recipe = window.fit_recipe.to_dict()
    monkeypatch.setattr(window, "_confirm_clear_fit_configuration", lambda exclusions: False)
    assert window.clear_fit_configuration() is False
    assert window.project.to_dict() == before_project
    assert window.fit_recipe.to_dict() == before_recipe
    assert window.solver_n_iter.value() == 7
    window.project.modified = False; window.close()


def test_fit_workflow_table_edits_sync_controls_and_persist(app, tmp_path, monkeypatch):
    window = MainWindow()
    window.parameter_checks["hbpm_gain"].setChecked(True)
    window.parameter_checks["skew_quads"].setChecked(False)
    window._store_active_fit_stage()
    window._duplicate_fit_stage()
    window.parameter_checks["hbpm_gain"].setChecked(False)
    window.parameter_checks["skew_quads"].setChecked(True)
    window.solver_n_iter.setValue(2)

    # Editable table cells update the FitStage source model and active controls.
    window.fit_workflow_table.item(1, 1).setText("Coupling correction")
    window.fit_workflow_table.item(1, 3).setText("5")
    assert window.fit_recipe.stages[1].name == "Coupling correction"
    assert window.solver_n_iter.value() == 5
    assert window._configuration_for_fit_stage(window.fit_recipe.stages[1]).solver.nIter == 5

    # Detailed controls update the table summary immediately.
    window.parameter_checks["skew_quads"].setChecked(False)
    window.parameter_checks["vbpm_coupling"].setChecked(True)
    assert "vbpm_coupling" in window.fit_workflow_table.item(1, 2).text()
    assert "skew_quads" not in window.fit_workflow_table.item(1, 2).text()
    window.solver_n_iter.setValue(6)
    assert window.fit_workflow_table.item(1, 3).text() == "6"
    window.loco_hor_dispersion_weight.setValue(2.5)
    assert "horizontal=2.5" in window.fit_workflow_table.item(1, 2).toolTip()
    window.bad_bpm_positions_edit.setText("0")
    window.bad_bpm_positions_edit.editingFinished.emit()
    assert "1 BPM" in window.fit_workflow_table.item(1, 2).toolTip()

    # Stage 1 cannot inherit; later stages expose the two valid choices.
    first_cell = window.fit_workflow_table.cellWidget(0, 4)
    second_cell = window.fit_workflow_table.cellWidget(1, 4)
    first_start = first_cell.findChild(QComboBox, "fitStageStartingModelEditor")
    second_start = second_cell.findChild(QComboBox, "fitStageStartingModelEditor")
    assert first_start is not None and second_start is not None
    assert window.fit_workflow_table.rowHeight(0) == 38
    assert window.fit_workflow_table.rowHeight(1) == 38
    assert first_start.maximumHeight() <= 32
    assert second_start.maximumHeight() <= 32
    assert first_cell.layout().contentsMargins().top() == 3
    assert second_cell.layout().contentsMargins().bottom() == 3
    assert first_start.minimumWidth() >= 245
    assert second_start.minimumWidth() >= 245
    assert first_start.currentData() == "original_model"
    assert first_start.isEnabled() is False
    assert [second_start.itemData(i) for i in range(second_start.count())] == [
        "original_model", "previous_stage"
    ]
    second_start.setCurrentIndex(second_start.findData("original_model"))
    assert window.fit_recipe.stages[1].start_from == "original_model"
    second_start = window.fit_workflow_table.cellWidget(1, 4).findChild(
        QComboBox, "fitStageStartingModelEditor"
    )
    second_start.setCurrentIndex(second_start.findData("previous_stage"))
    assert window.fit_recipe.stages[1].start_from == "previous_stage"

    # Invalid table edits are rejected without corrupting the model.
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append((args[1], args[2])))
    window.fit_workflow_table.item(1, 1).setText("   ")
    assert window.fit_recipe.stages[1].name == "Coupling correction"
    assert window.fit_workflow_table.item(1, 1).text() == "Coupling correction"
    window.fit_workflow_table.item(1, 3).setText("0")
    assert window.fit_workflow_table.item(1, 3).text() == "6"
    assert len(warnings) == 2

    # Project and standalone recipe round trips retain table edits.
    project_path = tmp_path / "table-edits.pyloco.json"
    recipe_path = tmp_path / "table-edits.recipe.json"
    window._snapshot_project_from_widgets()
    window.project.save(project_path)
    window.fit_recipe.save(recipe_path)
    restored_project = ProjectMetadata.load(project_path)
    restored_recipe = FitRecipe.load(recipe_path)
    assert restored_project.fit_recipe["stages"][1]["name"] == "Coupling correction"
    assert restored_project.fit_recipe["stages"][1]["configuration"]["backend_mapping"]["LOCOOptions"]["nIter"] == 6
    assert restored_recipe.stages[1].start_from == "previous_stage"
    assert restored_recipe.stages[1].name == "Coupling correction"

    # Reordering preserves valid inheritance and invalidates stale checkpoints.
    window.project.completed_run = CompletedRunReference(str(tmp_path / "old"), 1.0, "completed")
    window.project.fit_run_session = {"path": str(tmp_path / "session.json"), "completed_stages": 1}
    window._move_fit_stage(-1)
    assert window.fit_recipe.stages[0].start_from == "original_model"
    assert window.project.completed_run.results_dir == ""
    assert window.project.fit_run_session == {}
    assert window.project.compatibility_validation["status"] == "validation_required"
    window.project.modified = False; window.close()


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


def test_case_c_all_stages_keep_scientific_mapping_and_show_parameter_summary(app):
    import json
    from pathlib import Path

    project_path = Path(__file__).parents[1] / "Examples/EBS/multistage_fit/EBS_case_C_multistage.pyloco.json"
    source_stages = json.loads(project_path.read_text())["fit_recipe"]["stages"]
    window = MainWindow(); window.project.modified = False; window.open_project(project_path)
    app.processEvents()
    assert window.fit_workflow_table.rowCount() == 4
    assert window.fit_workflow_table.columnCount() == 5
    expected = (
        "hbpm_gain, vbpm_gain, hcor_cal, vcor_cal",
        "hbpm_coupling, vbpm_coupling", "quads",
        "skew_quads, hbpm_coupling, vbpm_coupling",
    )
    for index, summary in enumerate(expected):
        assert window.fit_workflow_table.item(index, 2).text() == summary
        window.fit_workflow_table.selectRow(index); app.processEvents()
        assert not window.constraint_details.isVisible()
        assert window.fixed_dk.text() == ""
        if index == 2:
            assert "initial ΔK = 0.001 × K" in window.normal_step_resolution.text()
        else:
            assert "not fitted" in window.normal_step_resolution.text()
    window._store_active_fit_stage()
    stored = window.fit_recipe.to_dict()["stages"]
    expected_mappings = [item["configuration"]["backend_mapping"] for item in source_stages]
    for mapping in expected_mappings:
        legacy = mapping["LOCOOptions"].get("remove_coupling_", True)
        mapping["LOCOOptions"]["orm_terms"] = "diagonal" if legacy else "full"
    assert [item["configuration"]["backend_mapping"] for item in stored] == expected_mappings
    window.project.modified = False; window.close()


def test_constraints_collapse_without_changing_values_and_init_literals_validate(app):
    window = MainWindow()
    window.constraint_quad_sigma.setValue(0.125)
    window.constraint_enabled.setChecked(False); app.processEvents()
    assert not window.constraint_details.isVisible()
    assert window.constraint_quad_sigma.value() == 0.125
    assert window._collect_loco_configuration().constraints.enable is False
    window.constraint_enabled.setChecked(True); app.processEvents()
    assert not window.constraint_details.isHidden()
    assert window.constraint_quad_sigma.value() == 0.125

    window.params_init.setText("{'hbpm_gain': 1.0}")
    assert window._collect_loco_configuration().parameters.init_value() == {"hbpm_gain": 1.0}
    window.params_init.setText("not a dictionary")
    with pytest.raises(ValueError, match="dictionary"):
        window._collect_loco_configuration().parameters.to_fit_init_config_kwargs()
    window.project.modified = False; window.close()


def test_advanced_mode_uses_the_guided_initial_value_dialog_not_raw_storage_window(app, monkeypatch):
    window = MainWindow()
    window.project.mode = "Advanced"
    window._apply_mode_visibility()
    app.processEvents()

    for raw_storage in (window.params_init, window.params_init_policy):
        assert raw_storage.parent() is window
        assert raw_storage.isHidden()
        assert not raw_storage.isWindow()

    captured = []
    monkeypatch.setattr(QDialog, "exec", lambda dialog: captured.append(dialog) or QDialog.Rejected)
    window._edit_explicit_initial_values()
    assert len(captured) == 1
    dialog = captured[0]
    assert dialog.objectName() == "explicitInitialValuesDialog"
    assert dialog.width() >= 560 and dialog.height() >= 300
    editor = dialog.findChild(QPlainTextEdit, "explicitInitialValuesEditor")
    assert editor is not None
    assert editor.placeholderText() == "{'hbpm_gain': 1.0}"
    help_label = dialog.findChild(QLabel, "explicitInitialValuesHelp")
    assert help_label is not None and "override the initialization policy" in help_label.text()
    buttons = dialog.findChild(QDialogButtonBox)
    assert buttons is not None
    assert buttons.button(QDialogButtonBox.Save) is not None
    assert buttons.button(QDialogButtonBox.Cancel) is not None
    window.project.modified = False
    window.close()


def test_fit_workflow_plain_language_and_fixed_circumference_control(app):
    window = MainWindow()
    assert window.fit_workflow_table.horizontalHeaderItem(2).text() == "Parameters fitted in this stage"
    assert window.fit_workflow_table.horizontalHeaderItem(4).text() == "Starting model"
    assert window.rm_fixedpath.text() == "fixedpathlength"
    assert window.loco_fixedpath.text() == "fixedpathlength"
    assert [window.orm_terms_to_fit.itemData(index) for index in range(window.orm_terms_to_fit.count())] == [
        "full", "diagonal", "off_diagonal"
    ]
    assert "model" in window.rm_coupling.text().lower()
    assert window.loco_plot_fit_parameters.text() == "Generate fitted-parameter plots"
    assert window.loco_fixedpath.isHidden()

    window.rm_fixedpath.setChecked(True); app.processEvents()
    assert window._collect_loco_configuration().response_matrix.fixedpathlength is True
    window.rm_fixedpath.setChecked(False); app.processEvents()
    assert window._collect_loco_configuration().response_matrix.fixedpathlength is False
    window.project.modified = False; window.close()


def test_orm_terms_are_stage_specific_and_analytical_coupling_fails_closed(app):
    window = MainWindow()
    window.orm_terms_to_fit.setCurrentIndex(window.orm_terms_to_fit.findData("diagonal"))
    window._store_active_fit_stage()
    window._add_fit_stage(name="Coupling", start_from="previous_stage")
    window.orm_terms_to_fit.setCurrentIndex(window.orm_terms_to_fit.findData("off_diagonal"))
    window._store_active_fit_stage()

    first = window._configuration_for_fit_stage(window.fit_recipe.stages[0])
    second = window._configuration_for_fit_stage(window.fit_recipe.stages[1])
    assert first.rejection.orm_terms == "diagonal"
    assert second.rejection.orm_terms == "off_diagonal"
    assert first.to_backend_mapping()["LOCOOptions"]["remove_coupling_"] is True
    assert second.to_backend_mapping()["LOCOOptions"]["remove_coupling_"] is False

    second.response_matrix.calculator = "Analytical"
    window.project.loco_config = second
    assert any("uncoupled" in message and "off-diagonal" in message for message in window.project.validation_messages())
    window.project.modified = False; window.close()


def test_reference_model_card_is_concise_but_retains_advanced_provenance(app):
    window=MainWindow()
    window.project.lattice.path=str(Path("Examples/EBS/multistage_fit/data/betamodel.mat").resolve())
    window._refresh_reference_model_info(); app.processEvents()
    text=window.reference_model_info.text()
    assert "Number of lattice elements" in text and ">4432<" in text
    assert "Slip factor η (pyAT convention)" in text
    assert "RF momentum coefficient" not in text
    assert "SHA-256" not in text and "Provenance" not in text
    assert "RF–revolution frequency consistency [Hz]" in text
    tooltip=window.reference_model_info.toolTip()
    assert "not a measured accelerator RF error" in tooltip
    assert "η_RF = αc − 1/γ² = −η_pyAT" in tooltip
    assert "SHA-256:" in tooltip and len(tooltip.split("SHA-256: ",1)[1].splitlines()[0]) == 64
    assert "Source / provenance: FIT reference lattice" in tooltip
    visible_labels=[label.text() for label in window.findChildren(QLabel) if label.isVisible()]
    assert "Elements" not in visible_labels
    window.project.modified=False; window.close()


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


@pytest.mark.parametrize("legacy_value,expected", [(True, "diagonal"), (False, "full")])
def test_legacy_remove_coupling_configuration_migrates_to_orm_terms(legacy_value, expected):
    configuration = LocoConfiguration.from_dict({
        "rejection": {"remove_coupling_": legacy_value},
    })
    assert configuration.rejection.orm_terms == expected
    backend = configuration.to_backend_mapping()["LOCOOptions"]
    assert backend["orm_terms"] == expected
    assert backend["remove_coupling_"] is legacy_value


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
    window.parameter_checks["quads"].setChecked(True)
    window.parameter_checks["skew_quads"].setChecked(True)
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
        if role_key in {"horizontal_corrector_ords", "vertical_corrector_ords"}:
            element.KickAngle = np.zeros(2)
        if role_key == "normal_quadrupole_ords":
            element.PolynomB = np.zeros(2)
        if role_key == "skew_quadrupole_ords":
            element.PolynomA = np.zeros(2)

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


def test_case_c_at_wildcard_selects_steering_sextupoles_with_exact_notebook_order(app):
    at = pytest.importorskip("at")
    lattice = at.load_lattice("Examples/EBS/multistage_fit/data/betamodel.mat")
    expected = np.asarray(at.get_refpts(lattice, "S[FIJ]2A*"), dtype=int).tolist()

    class Parent(QWidget):
        def _load_current_lattice(self):
            return lattice

    parent = Parent()
    dialog = ElementSelectionDialog(parent, "horizontal_corrector_ords", [])
    dialog.pattern_radio.setChecked(True)
    dialog.pattern_edit.setPlainText("S[FIJ]2A*")
    assert dialog._preview()
    assert dialog.selected_ords == expected
    assert len(expected) == dialog.table.rowCount() == 32
    assert dialog.selection_count.text() == "32 selected / 32 available · 32 visible · 0 highlighted"
    assert dialog.table.item(0, 1).text() == str(expected[0]) == "38"
    assert dialog.table.item(0, 2).text() == "SJ2A"
    assert dialog.table.item(0, 3).text() == "Sextupole"
    assert dialog.table.item(31, 2).text() == "SI2A"
    dialog.close(); parent.close()


def test_case_c_multiline_quad_wildcards_equal_notebook_combination(app):
    at = pytest.importorskip("at")
    lattice = at.load_lattice("Examples/EBS/multistage_fit/data/betamodel.mat")
    patterns = ("QD3[AE]*", "QF4[ABDE]*", "QD5[BD]*")
    expected = np.sort(np.concatenate([np.asarray(at.get_refpts(lattice, pattern), dtype=int) for pattern in patterns])).tolist()

    class Parent(QWidget):
        def _load_current_lattice(self):
            return lattice

    parent = Parent()
    dialog = ElementSelectionDialog(parent, "normal_quadrupole_ords", [])
    dialog.pattern_radio.setChecked(True)
    assert dialog.pattern_syntax.currentData() == "at_wildcard"
    dialog.pattern_edit.setPlainText("\n".join(patterns))
    assert dialog._preview()
    assert dialog.selected_ords == expected
    assert dialog.selected_ords == sorted(set(dialog.selected_ords))
    assert len(expected) == 252
    assert dialog.table.rowCount() == 252
    summary = dialog.pattern_match_summary.text()
    assert "QD3[AE]*: 62" in summary
    assert "QF4[ABDE]*: 126" in summary
    assert "QD5[BD]*: 64" in summary
    assert "combined: 252 unique elements" in summary
    dialog.close(); parent.close()


def test_multiline_patterns_deduplicate_report_empty_and_reject_invalid(app, monkeypatch):
    at = pytest.importorskip("at")
    lattice = at.load_lattice("Examples/EBS/multistage_fit/data/betamodel.mat")

    class Parent(QWidget):
        def _load_current_lattice(self):
            return lattice

    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: None)
    parent = Parent()
    dialog = ElementSelectionDialog(parent, "normal_quadrupole_ords", [])
    dialog.pattern_radio.setChecked(True)

    dialog.pattern_edit.setPlainText("QD3[AE]*\nQD3[AE]*")
    assert dialog._preview()
    assert len(dialog.selected_ords) == len(set(dialog.selected_ords)) == 62
    assert "QD3[AE]*: 62 · QD3[AE]*: 62 · combined: 62 unique elements" == dialog.pattern_match_summary.text()

    dialog.pattern_edit.setPlainText("NO_SUCH_FAMILY*")
    assert dialog._preview()
    assert dialog.selected_ords == []
    assert dialog.pattern_match_summary.text() == "NO_SUCH_FAMILY*: 0 · combined: 0 unique elements"

    dialog.pattern_edit.setPlainText("QD3[AE*")
    assert not dialog._preview()
    assert "Invalid AT wildcard" in dialog.validation_message.text()

    dialog.pattern_syntax.setCurrentIndex(dialog.pattern_syntax.findData("python_regex"))
    dialog.pattern_edit.setPlainText(r"^QD3[AE].*$")
    assert dialog._preview()
    assert len(dialog.selected_ords) == 62
    dialog.pattern_edit.setPlainText("[")
    assert not dialog._preview()
    assert "Invalid Python regular expression" in dialog.validation_message.text()
    dialog.close(); parent.close()


def test_case_c_skew_wildcard_accepts_multipole_hosts_with_polynoma_component(app, monkeypatch):
    at = pytest.importorskip("at")
    lattice = at.load_lattice("Examples/EBS/multistage_fit/data/betamodel.mat")

    class Parent(QWidget):
        def _load_current_lattice(self):
            return lattice

    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: None)
    parent = Parent()
    dialog = ElementSelectionDialog(parent, "skew_quadrupole_ords", [])
    dialog.pattern_radio.setChecked(True)
    dialog.pattern_edit.setPlainText("S[HFDIJ]*")

    notebook_ordinals = np.asarray(at.get_refpts(lattice, "S[HFDIJ]*"), dtype=int)
    assert dialog._preview()
    np.testing.assert_array_equal(np.asarray(dialog.selected_ords), notebook_ordinals)
    assert len(dialog.selected_ords) == 288
    assert all(len(np.asarray(lattice[ordinal].PolynomA)) > 1 for ordinal in dialog.selected_ords)
    assert "combined: 288 unique elements" in dialog.pattern_match_summary.text()
    assert "multipole" in dialog.table.item(0, 3).text().lower()
    dialog.close(); parent.close()


def test_skew_selection_rejects_multipole_without_required_polynoma_component(app, monkeypatch):
    class Multipole:
        FamName = "NO_SKEW_COMPONENT"
        PolynomA = np.asarray([0.0])

    class Parent(QWidget):
        def _load_current_lattice(self):
            return [Multipole()]

    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: None)
    parent = Parent()
    dialog = ElementSelectionDialog(parent, "skew_quadrupole_ords", [])
    dialog.manual_edit.setPlainText("0")
    assert not dialog._preview()
    assert "requires PolynomA[1]" in dialog.validation_message.text()
    assert "component is unavailable" in dialog.validation_message.text()
    dialog.close(); parent.close()


def test_element_selection_provenance_restores_patterns_and_manual_removal(app, monkeypatch):
    at = pytest.importorskip("at")
    lattice = at.load_lattice("Examples/EBS/multistage_fit/data/betamodel.mat")

    class Parent(QWidget):
        def _load_current_lattice(self):
            return lattice

    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: None)
    parent = Parent()
    patterns = "QD3[AE]*\nQF4[ABDE]*\nQD5[BD]*"
    first = ElementSelectionDialog(parent, "normal_quadrupole_ords", [])
    first.pattern_radio.setChecked(True)
    first.pattern_edit.setPlainText(patterns)
    assert first._preview()
    removed_ordinal = int(first.table.item(0, 0).data(Qt.UserRole))
    first.table.item(0, 0).setCheckState(Qt.Unchecked)
    first._accept_if_valid()

    assert first.selection_state["method"] == "at_wildcard"
    assert first.selection_state["patterns"] == patterns
    assert first.selection_state["manual_removals"] == [removed_ordinal]
    assert len(first.selection_state["generated_ords"]) == 252
    assert len(first.selected_ords) == 251

    reopened = ElementSelectionDialog(
        parent, "normal_quadrupole_ords", first.selected_ords, first.selection_state
    )
    assert reopened.pattern_radio.isChecked()
    assert reopened.pattern_syntax.currentData() == "at_wildcard"
    assert reopened.pattern_edit.toPlainText() == patterns
    assert reopened.selected_ords == first.selected_ords
    assert reopened.table.rowCount() == 252
    removed_row = next(
        row for row in range(reopened.table.rowCount())
        if int(reopened.table.item(row, 0).data(Qt.UserRole)) == removed_ordinal
    )
    assert reopened.table.item(removed_row, 0).checkState() == Qt.Unchecked
    assert "Manual removals: 1" in reopened.provenance_summary.text()
    reopened.close(); first.close(); parent.close()


def test_regex_provenance_and_lattice_mismatch_do_not_replace_stored_ordinals(app, monkeypatch):
    at = pytest.importorskip("at")
    lattice = at.load_lattice("Examples/EBS/multistage_fit/data/betamodel.mat")

    class Parent(QWidget):
        def _load_current_lattice(self):
            return lattice

    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: None)
    parent = Parent()
    stored = [int(value) for value in np.asarray(at.get_refpts(lattice, "QD3[AE]*"))]
    state = {
        "version": 1,
        "method": "python_regex",
        "patterns": r"^QD3[AE].*$",
        "generated_ords": stored[:-1],  # Simulate a different lattice/version resolution.
        "resolved_ords": stored[:-1],
        "manual_additions": [],
        "manual_removals": [],
    }
    dialog = ElementSelectionDialog(parent, "normal_quadrupole_ords", stored[:-1], state)
    assert dialog.pattern_syntax.currentData() == "python_regex"
    assert dialog.pattern_edit.toPlainText() == r"^QD3[AE].*$"
    assert dialog.selected_ords == stored[:-1]
    assert "Selection verification warning" in dialog.provenance_summary.text()
    assert "now resolves to 62 elements instead of the stored 61" in dialog.provenance_summary.text()
    dialog.close(); parent.close()


def test_legacy_element_selection_is_not_given_invented_provenance(app):
    class Quadrupole:
        FamName = "Q1"
        PolynomB = np.asarray([0.0, 1.0])

    class Parent(QWidget):
        def _load_current_lattice(self):
            return [Quadrupole()]

    parent = Parent()
    dialog = ElementSelectionDialog(parent, "normal_quadrupole_ords", [0], None)
    assert "Selection method not recorded — imported lattice indices." in dialog.provenance_summary.text()
    assert not any(button.isChecked() for button in (
        dialog.type_radio, dialog.pattern_radio, dialog.name_file_radio,
        dialog.file_radio, dialog.manual_radio,
    ))
    assert dialog.selected_ords == [0]
    dialog.close(); parent.close()


def test_corrector_selection_rejects_missing_steering_representation_with_actionable_error(app, monkeypatch):
    at = pytest.importorskip("at")
    lattice = at.Lattice([at.Drift("NOT_A_STEERER", 1.0)], energy=6.0e9)

    class Parent(QWidget):
        def _load_current_lattice(self):
            return lattice

    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: None)
    parent = Parent()
    dialog = ElementSelectionDialog(parent, "horizontal_corrector_ords", [])
    dialog.manual_edit.setPlainText("0")
    assert not dialog._preview()
    assert "no supported horizontal steering representation" in dialog.validation_message.text()
    assert "KickAngle or PolynomA/PolynomB" in dialog.validation_message.text()
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
        def __init__(self, _parent, _role_key, _current, _selection_state=None):
            self.selected_ords = [4, 5]
            self.selection_state = {
                "version": 1,
                "method": "manual_indices",
                "manual_text": "4, 5",
                "generated_ords": [4, 5],
                "resolved_ords": [4, 5],
                "manual_additions": [],
                "manual_removals": [],
            }

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


def test_parameter_selection_precedes_context_aware_jacobian_controls(app):
    window = MainWindow()
    fit_layout = window.parameter_group.parentWidget().layout()
    assert fit_layout.indexOf(window.parameter_group) < fit_layout.indexOf(window.jacobian_group)

    window.fixed_dk.setText("0.000234")
    window.fixed_delta_skew.setValue(0.000345)
    window.fixed_delta_q_tilt.setValue(0.000456)
    for key in ("quads", "skew_quads", "quads_tilt"):
        window.parameter_checks[key].setChecked(False)
    app.processEvents()
    assert not window.quad_jacobian_calculator.isEnabled()
    assert not window.skew_jacobian_calculator.isEnabled()
    assert not window.fixed_dk.isEnabled()
    assert not window.fixed_delta_skew.isEnabled()
    assert not window.fixed_delta_q_tilt.isEnabled()
    assert "Not used" in window.normal_step_resolution.text()
    assert window.skew_step_status.text() == "Not used in this stage"
    assert window.tilt_step_status.text() == "Not used in this stage"

    for key in ("quads", "skew_quads", "quads_tilt"):
        window.parameter_checks[key].setChecked(True)
    window.quad_jacobian_calculator.setCurrentText("Numerical")
    window.skew_jacobian_calculator.setCurrentText("Numerical")
    app.processEvents()
    assert window.quad_jacobian_calculator.isEnabled()
    assert window.skew_jacobian_calculator.isEnabled()
    assert window.fixed_dk.isEnabled()
    assert window.fixed_delta_skew.isEnabled()
    assert window.fixed_delta_q_tilt.isEnabled()
    assert window.fixed_dk.text() == "0.000234"
    assert window.fixed_delta_skew.value() == pytest.approx(0.000345)
    assert window.fixed_delta_q_tilt.value() == pytest.approx(0.000456)
    window.project.modified = False
    window.close()


def test_stage_switch_updates_parameter_and_jacobian_context(app):
    window = MainWindow()
    window.parameter_checks["quads"].setChecked(False)
    window.parameter_checks["skew_quads"].setChecked(True)
    window._store_active_fit_stage()
    window._duplicate_fit_stage()
    window.parameter_checks["quads"].setChecked(True)
    window.parameter_checks["skew_quads"].setChecked(False)
    window._store_active_fit_stage()

    window.fit_workflow_table.selectRow(0)
    app.processEvents()
    assert not window.quad_jacobian_calculator.isEnabled()
    assert window.skew_jacobian_calculator.isEnabled()
    window.fit_workflow_table.selectRow(1)
    app.processEvents()
    assert window.quad_jacobian_calculator.isEnabled()
    assert not window.skew_jacobian_calculator.isEnabled()
    window.project.modified = False
    window.close()


def test_momentum_compaction_source_visibility_and_persistence(app, tmp_path, monkeypatch):
    window = MainWindow()
    monkeypatch.setattr(window, "_automatic_momentum_compaction", lambda: (8.62361468e-5, None))
    window.mcf_source.setCurrentIndex(window.mcf_source.findData("automatic"))
    window._update_mcf_source_visibility()
    assert window.mcf_user_value.isHidden()
    assert window.mcf_user_value_label.isHidden()
    assert not window.mcf_automatic_summary.isHidden()
    assert "αc = 8.62361468e-05" in window.mcf_automatic_summary.text()
    assert "dimensionless" in window.mcf_automatic_summary.text()

    window.mcf_source.setCurrentIndex(window.mcf_source.findData("user"))
    window.mcf_user_value.setText("8.7e-5")
    app.processEvents()
    assert not window.mcf_user_value.isHidden()
    assert window.mcf_automatic_summary.isHidden()
    window.mcf_source.setCurrentIndex(window.mcf_source.findData("automatic"))
    assert window.mcf_user_value.text() == "8.7e-5"
    window.mcf_source.setCurrentIndex(window.mcf_source.findData("user"))
    window.project.loco_config = window._collect_loco_configuration()
    saved = window.project.save(tmp_path / "manual-mcf.pyloco.json")
    restored = ProjectMetadata.load(saved)
    assert restored.loco_config.mcf_source == "user"
    assert restored.loco_config.mcf_user_value == "8.7e-5"
    assert restored.loco_config.to_mcf_kwargs() == {"source": "user", "value": 8.7e-5}
    window.project.modified = False
    window.close()


def test_fit_import_export_actions_are_at_top_and_summary_remains_last(app):
    window = MainWindow()
    layout = window.fit_configuration_actions.parentWidget().layout()
    assert layout.indexOf(window.fit_configuration_actions) < layout.indexOf(
        window.fit_workflow_table.parentWidget()
    )
    assert layout.indexOf(window.summary_group) < layout.count() - 1  # final stretch follows it
    buttons = window.fit_configuration_actions.findChildren(main_window_module.QPushButton)
    labels = [button.text() for button in buttons]
    assert labels.count("Import configuration…") == 1
    assert labels.count("Export configuration…") == 1
    assert labels.count("Review configuration") == 1
    window.project.modified = False
    window.close()


def test_fit_summary_separates_current_all_stages_and_backend_mapping(app):
    window = MainWindow()
    window.parameter_checks["quads"].setChecked(False)
    window.parameter_checks["hbpm_gain"].setChecked(True)
    window._store_active_fit_stage()
    window._duplicate_fit_stage()
    window.parameter_checks["hbpm_gain"].setChecked(False)
    window.parameter_checks["skew_quads"].setChecked(True)
    window._on_fit_config_changed()
    app.processEvents()

    assert "Current stage overview" in window.fit_current_stage_summary.text()
    assert "Stage 2" in window.fit_current_stage_summary.text()
    all_stages = window.fit_all_stages_summary.text()
    assert "All stages overview" in all_stages
    assert window.fit_recipe.stages[0].name in all_stages
    assert window.fit_recipe.stages[1].name in all_stages
    readable = window.fit_scientific_summary.text()
    assert "Inactive; 0 active fitting parameters" in readable
    assert "physical magnets selected" in readable
    expected = json.dumps(
        json_safe(window.project.loco_config.to_backend_mapping()), indent=2
    )
    assert window.fit_backend_options.toPlainText() == expected
    assert not window.fit_backend_group.isChecked()
    assert window.fit_backend_contents.isHidden()

    window.fit_workflow_table.selectRow(0)
    app.processEvents()
    assert "Stage 1" in window.fit_current_stage_summary.text()
    window.project.modified = False
    window.close()


def test_import_configuration_confirms_scope_and_preserves_recipe(app, tmp_path, monkeypatch):
    window = MainWindow()
    original_stage_names = [stage.name for stage in window.fit_recipe.stages]
    imported = ProjectMetadata().loco_config
    imported.solver.nIter = 7
    source = imported.save(tmp_path / "fit-config.json")
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *_args, **_kwargs: (str(source), ""))
    )
    prompts = []

    def cancel(_parent, title, message, *_args, **_kwargs):
        prompts.append((title, message))
        return QMessageBox.Cancel

    monkeypatch.setattr(QMessageBox, "question", cancel)
    window.import_loco_configuration()
    assert window.solver_n_iter.value() != 7
    assert "does not load a multi-stage recipe" in prompts[0][1]
    assert "Keep the current reference lattice" in prompts[0][1]

    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.Yes)
    window.import_loco_configuration()
    assert window.solver_n_iter.value() == 7
    assert [stage.name for stage in window.fit_recipe.stages] == original_stage_names
    stage_mapping = window.fit_recipe.stages[window._active_fit_stage].configuration["backend_mapping"]
    assert stage_mapping["LOCOOptions"]["nIter"] == 7
    window.project.modified = False
    window.close()


def test_export_configuration_from_moved_action_preserves_backend_mapping(app, tmp_path, monkeypatch):
    window = MainWindow()
    window.solver_n_iter.setValue(4)
    expected = window._collect_loco_configuration().to_backend_mapping()
    target = tmp_path / "exported-fit-config.json"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *_args, **_kwargs: (str(target), ""))
    )
    window.export_loco_configuration()
    assert target.is_file()
    exported, measurements, lattice = main_window_module.load_example_project_data(target)
    assert measurements == {}
    assert not lattice
    assert exported.to_backend_mapping() == expected
    window.project.modified = False
    window.close()


def test_remaining_case_c_stage_edits_match_table_recipe_project_and_request(app, tmp_path):
    source = Path(__file__).resolve().parents[1] / "Examples/EBS/multistage_fit/EBS_case_C_multistage.pyloco.json"
    project_copy = tmp_path / "case-c-copy.pyloco.json"
    project_copy.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    shutil.copytree(source.parent / "data", tmp_path / "data")
    window = MainWindow()
    window.open_project(project_copy)
    app.processEvents()

    while len(window.fit_recipe.stages) > 1:
        window.fit_workflow_table.selectRow(0)
        app.processEvents()
        window._remove_fit_stage()
        app.processEvents()
    assert window.fit_recipe.stages[0].name == "Skew and coupling"

    calibration = {"hbpm_gain", "vbpm_gain", "hcor_cal", "vcor_cal"}
    for key, checkbox in window.parameter_checks.items():
        checkbox.setChecked(key in calibration)
    app.processEvents()
    expected = ["hbpm_gain", "vbpm_gain", "hcor_cal", "vcor_cal"]
    assert window.fit_workflow_table.item(0, 2).text() == ", ".join(expected)
    stage_mapping = window.fit_recipe.stages[0].configuration["backend_mapping"]
    assert stage_mapping["FitInitConfig"]["fit_list"] == expected
    assert window.fit_recipe.to_dict()["stages"][0]["configuration"]["backend_mapping"]["FitInitConfig"]["fit_list"] == expected
    assert window.project.loco_config.to_backend_mapping()["FitInitConfig"]["fit_list"] == expected
    assert main_window_module.LocoRunRequest.from_project(window.project).backend_mapping["FitInitConfig"]["fit_list"] == expected
    window._validate_active_stage_widget_consistency()

    window.project.fit_recipe = window.fit_recipe.to_dict()
    saved = window.project.save(tmp_path / "edited-case-c.pyloco.json")
    reopened = ProjectMetadata.load(saved)
    reopened_stage = reopened.fit_recipe["stages"][0]["configuration"]["backend_mapping"]
    assert reopened_stage["FitInitConfig"]["fit_list"] == expected

    # A synthetic divergence is rejected instead of being silently repaired at launch.
    window.fit_recipe.stages[0].configuration["backend_mapping"]["FitInitConfig"]["fit_list"] = ["skew_quads"]
    with pytest.raises(ValueError, match="visible Parameter Selection"):
        window._validate_active_stage_widget_consistency()
    window.project.modified = False
    window.close()
