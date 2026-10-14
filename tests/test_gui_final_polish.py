from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QDialog, QDialogButtonBox, QFileDialog,
    QLabel, QMessageBox, QPlainTextEdit, QWidget,
)

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
    assert [item["configuration"]["backend_mapping"] for item in stored] == [
        item["configuration"]["backend_mapping"] for item in source_stages
    ]
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
    assert window.loco_remove_coupling.text() == "Exclude cross-plane ORM responses from the fit"
    assert window.loco_plot_fit_parameters.text() == "Generate fitted-parameter plots"
    assert window.loco_fixedpath.isHidden()

    window.rm_fixedpath.setChecked(True); app.processEvents()
    assert window._collect_loco_configuration().response_matrix.fixedpathlength is True
    window.rm_fixedpath.setChecked(False); app.processEvents()
    assert window._collect_loco_configuration().response_matrix.fixedpathlength is False
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
    assert dialog.selection_count.text() == "32 selected / 32 in preview"
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
