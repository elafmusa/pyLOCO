from __future__ import annotations

import json
import os
from copy import deepcopy
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QAbstractButton, QFileDialog, QLabel, QMessageBox, QScrollArea, QToolButton
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence

from pyLOCO.gui.backend import (
    _apply_bad_bpm_positions,
    _apply_corrector_exclusions,
    _apply_machine_element_selections,
    _build_pyloco_kwargs,
    _assemble_measured_response,
    _load_measurements,
    _save_jacobian,
    _save_optics,
)
from pyLOCO.gui.machine_detection import detect_machine_elements
from pyLOCO.gui.measurement_metadata import inspect_measurement_metadata
from pyLOCO.gui.models.project import CompletedRunReference, ImportedDataset, LocoConfiguration, ProjectMetadata
from pyLOCO.gui.main_window import ElementSelectionDialog, ExclusionSelectionDialog, MainWindow, ScientificDoubleSpinBox, ScrollSafeSpinBox
from pyLOCO.gui.fit_workflow import FitStage
from pyLOCO.gui.measurement_mapping import find_exact_device_mappings
from pyLOCO.gui.results.parameters_view import ParametersView
from pyLOCO.gui.results.plot_canvas import PlotCanvas
from pyLOCO.gui.results.results_loader import ResultsLoader
from pyLOCO.gui.results.results_workspace import ResultsWorkspace
from pyLOCO.gui.results.run_summary_view import RunSummaryView


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_reference_lattice_change_invalidates_saved_mapping_evidence(app):
    window = MainWindow()
    window.project.measurements["orm"] = ImportedDataset("orm", "measurement.h5", "h5", 1, {})
    window.project.lattice.path = "new-reference.mat"
    window.project.measurement_device_mapping = {
        "mappings": [{"role": "bpm", "measured_name": "BPM1", "status": "mapped"}],
    }
    window.project.machine_mapping = {"schema": "pyloco.machine_mapping", "mappings": [{"role": "bpm"}]}
    window.project.loco_config.element_selection_state["bpm_ords"] = {
        "method": "validated_measurement_mapping", "lattice_sha256": "old",
    }
    window.project.loco_config.element_selection_state["normal_quadrupole_ords"] = {
        "method": "manual", "resolved_ords": [1],
    }

    window._invalidate_mapping_for_reference_lattice_change()

    assert window.project.measurement_device_mapping == {}
    assert window.project.machine_mapping == {}
    assert "bpm_ords" not in window.project.loco_config.element_selection_state
    assert window.project.loco_config.element_selection_state["normal_quadrupole_ords"]["method"] == "manual"
    assert window.use_measurement_mappings_button.isEnabled() is False
    assert "No validated mapping exists for the active reference lattice" in window.measurement_mapping_feedback.text()
    window.project.modified = False
    window.close()


def test_incomplete_mapping_opens_actionable_measurement_rows(app, monkeypatch):
    window = MainWindow()
    mapping = {
        "mappings": [
            {"role": "bpm", "measured_name": "BPM1", "status": "mapped"},
            {"role": "horizontal_corrector", "measured_name": "CH1", "status": "unresolved",
             "reason": "No exact identity match"},
        ],
    }
    monkeypatch.setattr(window, "_measurement_mapping_context", lambda: (
        [], "reference.mat", {}, {"bpm": ["BPM1"], "horizontal_corrector": ["CH1"],
                                   "vertical_corrector": []}, None,
    ))
    monkeypatch.setattr("pyLOCO.gui.main_window.find_exact_device_mappings", lambda *_args, **_kwargs: mapping)
    monkeypatch.setattr(window, "_find_complete_machine_mapping", lambda *_args: {})

    assert window.find_measurement_mappings() is False
    assert window.mapping_table_scope.currentData() == "measurement"
    assert window.measurement_mapping_filter.currentData() == "attention"
    assert window.measurement_mapping_table.rowCount() == 1
    assert window.measurement_mapping_table.item(0, 0).text() == "CH1"
    window.project.modified = False
    window.close()


def test_loading_mapping_for_old_lattice_offers_safe_rebuild(app, tmp_path, monkeypatch):
    old_lattice = tmp_path / "old.mat"
    current_lattice = tmp_path / "current.mat"
    old_lattice.write_bytes(b"old lattice")
    current_lattice.write_bytes(b"current lattice")
    inventory = {"bpm": [], "horizontal_corrector": [], "vertical_corrector": []}
    document = find_exact_device_mappings([], inventory, lattice_path=old_lattice)
    mapping_path = tmp_path / "old-mapping.json"
    mapping_path.write_text(json.dumps(document), encoding="utf-8")

    window = MainWindow()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args, **_kwargs: (str(mapping_path), ""))
    monkeypatch.setattr(window, "_measurement_mapping_context", lambda: (
        [], current_lattice, {}, inventory, None,
    ))
    rebuilt = []
    def rebuild():
        rebuilt.append(True)
        window.project.measurement_device_mapping = {
            "mappings": [{"role": "bpm", "measured_name": "BPM1", "status": "unresolved"}],
        }
        return False
    monkeypatch.setattr(window, "find_measurement_mappings", rebuild)

    class RebuildDialog:
        Warning, AcceptRole, RejectRole = 1, 2, 3
        notices = []
        def __init__(self, *_args): self.buttons = []
        def setWindowTitle(self, *_args): pass
        def setIcon(self, *_args): pass
        def setText(self, *_args): pass
        def setInformativeText(self, *_args): pass
        def addButton(self, label, *_args):
            button = object(); self.buttons.append((label, button)); return button
        def setDefaultButton(self, *_args): pass
        def exec(self): return 0
        def clickedButton(self): return self.buttons[0][1]
        @staticmethod
        def information(*args, **_kwargs): RebuildDialog.notices.append(args)
        @staticmethod
        def warning(*args, **_kwargs): RebuildDialog.notices.append(args)

    monkeypatch.setattr("pyLOCO.gui.main_window.QMessageBox", RebuildDialog)
    assert window.load_measurement_mapping() is False
    assert rebuilt == [True]
    assert RebuildDialog.notices
    assert "1 of 1 measured devices matched; 0 remain unresolved" not in RebuildDialog.notices[0][-1]
    assert "0 of 1 measured devices matched; 1 remain unresolved" in RebuildDialog.notices[0][-1]
    window.project.modified = False
    window.close()


def test_use_validated_measurement_mapping_applies_exact_order_to_all_stages_and_gates_run(app, tmp_path, monkeypatch):
    class BPM:
        def __init__(self, name, device): self.FamName, self.Device = name, device
    class Multipole:
        def __init__(self, name, device):
            self.FamName, self.Device = name, device
            self.PolynomA, self.PolynomB = [0.0, 0.0], [0.0, 0.0]
            self.Length, self.PassMethod = 1.0, "StrMPoleSymplectic4Pass"

    lattice_path = tmp_path / "ring.mat"; lattice_path.write_bytes(b"mapping-order-lattice")
    ring = [BPM("B1", "bpm/1"), BPM("B2", "bpm/2"),
            Multipole("S1", "steerer/1"), Multipole("S2", "steerer/2")]
    # Deliberately non-sorted BPM identity order proves that the proposal does
    # not sort AT ordinals or infer identity from array position.
    inventory = {"bpm": ["bpm/2", "bpm/1"],
                 "horizontal_corrector": ["steerer/2-B1L"],
                 "vertical_corrector": ["steerer/1-A1L"]}
    mapping = find_exact_device_mappings(ring, inventory, lattice_path=lattice_path)
    window = MainWindow()
    window.project.lattice.path = str(lattice_path)
    window.project.measurement_device_mapping = mapping
    elements = window.project.loco_config.machine_elements
    elements.bpm_ords = [0, 1, 9]; elements.horizontal_corrector_ords = [2, 3]
    elements.vertical_corrector_ords = [3, 2]; elements.normal_quadrupole_ords = [77]
    window.project.loco_config.bad_bpm_positions = [0]
    window.project.completed_run = CompletedRunReference(str(tmp_path / "old-results"), 1.0, "completed")
    window.project.fit_run_session = {"path": str(tmp_path / "old-session.json"), "completed_stages": 1}
    window._load_config_to_widgets()
    first = window.fit_recipe.stages[0]
    first.configuration = window._stage_configuration_snapshot()
    second = FitStage(name="Stage 2", configuration=__import__("copy").deepcopy(first.configuration),
                      start_from="previous_stage")
    second.configuration["backend_mapping"]["MachineElements"]["bpm_ords"] = [1]
    second.configuration["gui_config"]["machine_elements"]["bpm_ords"] = [1]
    window.fit_recipe.stages.append(second)
    monkeypatch.setattr(window, "_measurement_mapping_context", lambda: (
        ring, lattice_path, {"files": {"orm": {"shape": [4, 2]}}}, inventory, None,
    ))
    proposal = window._validated_mapping_fit_proposal()
    assert proposal["proposed"] == {
        "bpm_ords": [1, 0], "horizontal_corrector_ords": [3], "vertical_corrector_ords": [2],
    }
    assert proposal["expected_shape"] == (4, 2)
    assert proposal["distinct_stage_selections"] == 2
    assert proposal["distinct_stage_exclusions"] == 1
    assert proposal["preserve_exclusions"] is False
    before_cancel = window.project.to_dict()
    monkeypatch.setattr(window, "_confirm_mapping_fit_proposal", lambda value: False)
    assert window.use_measurement_mappings_for_fit() is False
    assert window.project.to_dict() == before_cancel
    monkeypatch.setattr(window, "_confirm_mapping_fit_proposal", lambda value: True)
    assert window.use_measurement_mappings_for_fit() is True
    selected = window.project.loco_config.machine_elements
    assert selected.bpm_ords == [1, 0]
    assert selected.horizontal_corrector_ords == [3]
    assert selected.vertical_corrector_ords == [2]
    assert selected.normal_quadrupole_ords == [77]
    assert window.project.loco_config.bad_bpm_positions == []
    assert window.project.loco_config.element_selection_state["bpm_ords"]["method"] == "validated_measurement_mapping"
    for stage in window.fit_recipe.stages:
        cfg = window._configuration_for_fit_stage(stage)
        assert cfg.machine_elements.bpm_ords == [1, 0]
        assert cfg.machine_elements.horizontal_corrector_ords == [3]
        assert cfg.machine_elements.vertical_corrector_ords == [2]
    assert window.project.completed_run.results_dir == ""
    assert window.project.fit_run_session == {}
    assert window.project.compatibility_validation["status"] == "validation_required"
    assert window.run_loco_action.isEnabled() is False
    saved = window.project.save(tmp_path / "mapped-adapted.pyloco.json")
    restored = ProjectMetadata.load(saved)
    assert restored.loco_config.machine_elements.bpm_ords == [1, 0]
    assert restored.compatibility_validation["status"] == "validation_required"
    window.project.modified = False; window.close()


def test_measurement_mapping_guidance_tracks_empty_partial_full_and_applied_states(app, tmp_path):
    window = MainWindow()
    window.project.measurement_device_mapping = {}
    window._refresh_measurement_mapping_feedback()
    assert "load measurements" in window.measurement_mapping_feedback.text().lower()
    assert not window.use_measurement_mappings_button.isEnabled()
    window.project.measurement_device_mapping = {"mappings": [
        {"status": "mapped"}, {"status": "unresolved"},
    ]}
    window._refresh_measurement_mapping_feedback()
    assert "mapping incomplete" in window.measurement_mapping_feedback.text().lower()
    assert not window.use_measurement_mappings_button.isEnabled()
    window.project.measurement_device_mapping = {"mappings": [
        {"status": "mapped"}, {"status": "mapped"},
    ]}
    window._refresh_measurement_mapping_feedback()
    assert "next: use mappings" in window.measurement_mapping_feedback.text().lower()
    assert window.use_measurement_mappings_button.isEnabled()
    for field in ("bpm_ords", "horizontal_corrector_ords", "vertical_corrector_ords"):
        window.project.loco_config.element_selection_state[field] = {"method": "validated_measurement_mapping"}
    window._refresh_measurement_mapping_feedback()
    assert "next: validate" in window.measurement_mapping_feedback.text().lower()
    window.project.modified = False; window.close()


def test_recent_project_path_can_be_copied_with_button_or_standard_copy_action(app, tmp_path):
    path = str(tmp_path / "Project with spaces.pyloco.json")
    window = MainWindow()
    window.project.recent_projects = [path]
    window._refresh_ui("recent project")
    assert window.copy_recent_path_button.isEnabled() is False
    window.recent_list.setCurrentRow(0)
    assert window.copy_recent_path_button.isEnabled() is True
    QApplication.clipboard().clear()
    assert window._copy_selected_recent_project_path() is True
    assert QApplication.clipboard().text() == path
    copy_actions = [action for action in window.recent_list.actions() if action.shortcut() == QKeySequence.Copy]
    assert len(copy_actions) == 1
    QApplication.clipboard().clear()
    copy_actions[0].trigger()
    assert QApplication.clipboard().text() == path
    window.project.modified = False; window.close()


def test_project_state_round_trip_includes_exclusions_selection_and_run(tmp_path):
    project = ProjectMetadata(name="round trip")
    project.loco_config.bad_bpm_positions = [1, 4]
    project.loco_config.excluded_horizontal_corrector_positions = [2]
    project.loco_config.excluded_vertical_corrector_positions = [3]
    project.loco_config.element_selection_state = {"cavity_ords": {"method": "automatic"}}
    project.loco_config.rejection.includeDispersion = True
    project.loco_config.rejection.hor_dispersion_weight = 2.5
    project.loco_config.rejection.ver_dispersion_weight = 3.5
    result_dir = tmp_path / "results"; result_dir.mkdir()
    project.completed_run = CompletedRunReference(str(result_dir), 12.5, "completed")
    path = project.save(tmp_path / "state.pyloco.json")
    restored = ProjectMetadata.load(path)
    assert restored.loco_config.bad_bpm_positions == [1, 4]
    assert restored.loco_config.excluded_horizontal_corrector_positions == [2]
    assert restored.loco_config.element_selection_state["cavity_ords"]["method"] == "automatic"
    assert restored.loco_config.rejection.hor_dispersion_weight == 2.5
    assert restored.completed_run.results_dir == str(result_dir)


def test_old_project_configuration_remains_compatible():
    restored = LocoConfiguration.from_dict({"bad_bpm_positions": [0]})
    assert restored.bad_bpm_positions == [0]
    assert restored.excluded_horizontal_corrector_positions == []


def test_measurement_mapping_review_defaults_to_attention_and_filters(app):
    window = MainWindow()
    window.project.measurement_device_mapping = {
        "mappings": [
            {"measured_name": "machine/bpm/1", "role": "bpm", "lattice_name": "BPM1",
             "lattice_ordinal": 10, "component": "", "status": "mapped",
             "reason": "Source and AT identity revalidated"},
            {"measured_name": "machine/hcor/1-B1L", "role": "horizontal_corrector",
             "lattice_name": "", "lattice_ordinal": None, "component": "B1L",
             "status": "unresolved", "reason": "No exact Device identity"},
        ]
    }
    window._refresh_measurement_mapping_table()
    assert window.measurement_mapping_filter.currentData() == "attention"
    assert window.measurement_mapping_table.rowCount() == 1
    assert window.measurement_mapping_table.item(0, 0).text() == "machine/hcor/1-B1L"
    window.measurement_mapping_filter.setCurrentIndex(
        window.measurement_mapping_filter.findData("all")
    )
    assert window.measurement_mapping_table.rowCount() == 2
    window.measurement_mapping_search.setText("BPM1")
    assert window.measurement_mapping_table.rowCount() == 1
    assert window.save_measurement_mapping_button.text() == "Save validated Machine Mapping…"
    window.close()


def test_machine_configuration_precedes_manual_elements_and_has_intro(app):
    window = MainWindow()
    layout = window.machine_config_group.parentWidget().layout()
    assert layout.indexOf(window.machine_config_group) < layout.indexOf(window.machine_elements_group)
    assert window.machine_lattice_intro.text() == (
        "Load or select machine elements for LOCO. BPMs and correctors define the measured devices; "
        "exclusions remove selected devices from fitting. Mapping links measured devices to AT lattice elements."
    )
    window.close()


def test_completed_mapping_operation_shows_all_records_and_preserves_manual_filter(app):
    window = MainWindow()
    mapping = {
        "mappings": [
            {"measured_name": f"machine/bpm/{index}", "role": "bpm", "lattice_name": f"BPM{index}",
             "lattice_ordinal": index, "component": "", "status": "mapped", "reason": "Exact identity"}
            for index in range(330)
        ]
    }
    window.project.measurement_device_mapping = mapping
    window._select_measurement_mapping_filter_after_operation(mapping)
    window._refresh_measurement_mapping_table()
    assert window.measurement_mapping_filter.currentData() == "all"
    assert window.measurement_mapping_table.rowCount() == 330
    assert window.measurement_mapping_table_hint.isHidden()

    window.measurement_mapping_filter.setCurrentIndex(
        window.measurement_mapping_filter.findData("validated")
    )
    window._refresh_measurement_mapping_table()
    window._refresh_measurement_mapping_feedback()
    assert window.measurement_mapping_filter.currentData() == "validated"
    assert window.measurement_mapping_table.rowCount() == 330
    window.close()


def test_incomplete_mapping_operation_defaults_to_attention_and_empty_state_is_explained(app):
    window = MainWindow()
    assert window.measurement_mapping_table.rowCount() == 0
    assert "No mapping records yet" in window.measurement_mapping_table_hint.text()

    mapping = {
        "mappings": [
            {"measured_name": "machine/bpm/1", "role": "bpm", "lattice_name": "BPM1",
             "lattice_ordinal": 1, "component": "", "status": "mapped", "reason": "Exact identity"},
            {"measured_name": "machine/hcor/1", "role": "horizontal_corrector", "lattice_name": "",
             "lattice_ordinal": None, "component": "B1L", "status": "unresolved", "reason": "No identity"},
        ]
    }
    window.project.measurement_device_mapping = mapping
    window._select_measurement_mapping_filter_after_operation(mapping)
    window._refresh_measurement_mapping_table()
    assert window.measurement_mapping_filter.currentData() == "attention"
    assert window.measurement_mapping_table.rowCount() == 1
    assert window.measurement_mapping_table.item(0, 1).text() == "H corrector"
    assert window.measurement_mapping_table.item(0, 6).text() == "B1L — not validated"
    window.close()


def test_measurement_metadata_propagation_values(tmp_path):
    path = tmp_path / "orm.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("response_matrix", data=np.zeros((4, 2)))
        handle.attrs["dkick_rad"] = 1e-4
        handle.attrs["bidirectional"] = True
    metadata = inspect_measurement_metadata(path, "orm")
    assert metadata["dkick_h"] == pytest.approx(1e-4)
    assert metadata["dkick_v"] == pytest.approx(1e-4)
    assert metadata["bidirectional"] is True


def test_measurement_metadata_mismatch_is_non_blocking_warning(tmp_path):
    project = ProjectMetadata()
    project.measurements["orm"] = ImportedDataset("orm", str(tmp_path / "orm.h5"), "h5", options={"dkick_h": 1e-4})
    project.loco_config.response_matrix.dkick_h = 1e-5
    assert any("horizontal ORM kick" in item for item in project.metadata_warnings())


def test_corrector_exclusion_removes_matching_columns_and_ordinals():
    measured = {"orm": np.arange(24).reshape(4, 6)}
    indices = {"nHorCOR": 3, "nVerCOR": 3, "used_cor_ords": [np.array([10, 11, 12]), np.array([20, 21, 22])]}
    updated, new_indices, steps = _apply_corrector_exclusions(measured, indices, [1], [0], ([1., 2., 3.], [4., 5., 6.]))
    assert updated["orm"].shape == (4, 4)
    assert new_indices["used_cor_ords"][0].tolist() == [10, 12]
    assert new_indices["used_cor_ords"][1].tolist() == [21, 22]
    assert np.asarray(steps[0]).tolist() == [1., 3.]


def test_bad_bpm_positions_apply_to_explicit_selected_bpm_list():
    class RM: bpm_ords = []; cm_ords = ([], []); cav_ords = []
    measured = {"orm": np.zeros((6, 2)), "noise_x": np.ones(3), "noise_y": np.ones(3),
                "eta_x": np.zeros(3), "eta_y": np.zeros(3)}
    indices = {"used_bpms_ords": np.array([1, 2, 3]), "used_cor_ords": [np.array([4]), np.array([5])],
               "nHBPM": 3, "nVBPM": 3, "nHorCOR": 1, "nVerCOR": 1}
    indices = _apply_machine_element_selections(indices, {"bpm_ords": [10, 20, 30]}, RM())
    def remove_rows(matrix, positions, **_kwargs):
        rows = np.concatenate((positions, 3 + positions))
        return np.delete(matrix, rows, axis=0), None
    _measured, updated = _apply_bad_bpm_positions(measured, indices, np.array([1]), remove_rows)
    assert updated["used_bpms_ords"].tolist() == [10, 30]


def test_rf_cavity_detection_uses_at_definition():
    at = pytest.importorskip("at")
    ring = at.Lattice([at.Drift("D", 1.0), at.RFCavity("not_named_like_rf", 0.0, 3e6, 5e8, 992, 3e9)])
    assert detect_machine_elements(ring, "cavity") == [1]


def test_selection_mode_only_enables_relevant_controls(app):
    window = MainWindow(); dialog = ElementSelectionDialog(window, "bpm_ords", [])
    assert not hasattr(dialog,"auto_radio")
    assert dialog.manual_edit.isVisibleTo(dialog)
    assert not dialog.type_edit.isVisibleTo(dialog) and not dialog.pattern_edit.isVisibleTo(dialog)
    dialog.name_file_radio.setChecked(True)
    assert dialog.name_file_edit.isVisibleTo(dialog) and dialog.name_attribute.isVisibleTo(dialog)
    assert not dialog.file_edit.isVisibleTo(dialog) and not dialog.pattern_edit.isVisibleTo(dialog)
    window.close()


@pytest.mark.parametrize("width,height", [(1000, 700), (1200, 800), (1500, 900)])
def test_machine_component_rows_remain_aligned_and_scrollable(app, width, height):
    window = MainWindow()
    window._workspace.setCurrentIndex(1)
    window.resize(width, height)
    window.show()
    app.processEvents()

    rows = list(window.element_row_widgets.values())
    buttons = list(window.element_edit_buttons.values())
    # Quadrupole tilts now have an independent physical-element selector.
    assert len(rows) == len(buttons) == 7
    assert len({(button.width(), button.height()) for button in buttons}) == 1
    assert all(button.width() >= button.sizeHint().width() for button in buttons)
    for previous, current in zip(rows, rows[1:]):
        assert previous.geometry().bottom() < current.geometry().top()
        assert previous.height() >= 44 and current.height() >= 44

    scroll = window.findChild(QScrollArea, "machineComponentsScroll")
    assert scroll is not None
    if (width, height) == (1000, 700):
        assert scroll.verticalScrollBar().maximum() > 0
    assert window.bad_bpm_positions_edit.height() >= 34
    assert window.hcor_exclusions_edit.height() >= 34
    assert window.vcor_exclusions_edit.height() >= 34
    assert window.exclusion_counts.wordWrap()
    window.close()


def test_spinbox_wheel_never_changes_values_while_scrolling(app):
    class Event:
        ignored = False
        def ignore(self): self.ignored = True
    for spin in (ScientificDoubleSpinBox(),ScrollSafeSpinBox()):
        event = Event(); spin.setFocus(); spin.wheelEvent(event)
        assert event.ignored


def test_machine_element_and_exclusion_guidance_uses_selected_list_positions(app):
    window = MainWindow()
    assert "Edit/Select to check or uncheck individual elements" in window.machine_elements_help.text()
    assert "Normal and skew quadrupoles" in window.machine_elements_help.toolTip()
    assert "no separate quadrupole-exclusion field" in window.machine_elements_help.toolTip()
    assert "zero-based positions" in window.exclusion_help.text()
    assert "not AT lattice ordinals" in window.exclusion_help.text()
    assert "BPM position 27 is the 28th BPM" in window.exclusion_help.text()
    assert "keep the original selected device list unchanged" in window.exclusion_help.toolTip()
    for editor in (
        window.bad_bpm_positions_edit,
        window.hcor_exclusions_edit,
        window.vcor_exclusions_edit,
    ):
        assert "not AT lattice ordinals" in editor.toolTip()
    window.close()


def test_dispersion_weights_are_contextual_preserved_and_routed(app):
    window = MainWindow()
    window.show()
    window.rm_dispersion.setChecked(False)
    app.processEvents()
    assert window.dispersion_weight_controls.isHidden()
    assert not window.loco_hor_dispersion_weight.isVisible()
    assert not window.loco_ver_dispersion_weight.isVisible()

    window.rm_dispersion.setChecked(True)
    window.loco_hor_dispersion_weight.setValue(5.0)
    window.loco_ver_dispersion_weight.setValue(2.5)
    app.processEvents()
    assert not window.dispersion_weight_controls.isHidden()
    assert window.loco_hor_dispersion_weight.isEnabled()
    assert window.loco_ver_dispersion_weight.isEnabled()

    window.rm_dispersion.setChecked(False)
    window.rm_dispersion.setChecked(True)
    app.processEvents()
    assert window.loco_hor_dispersion_weight.value() == 5.0
    assert window.loco_ver_dispersion_weight.value() == 2.5

    config = window._collect_loco_configuration()
    assert config.response_matrix.includeDispersion is True
    assert config.rejection.includeDispersion is True
    assert config.rejection.hor_dispersion_weight == 5.0
    assert config.rejection.ver_dispersion_weight == 2.5

    visible_text = [
        widget.text()
        for widget in (
            window.findChildren(QLabel) + window.findChildren(QAbstractButton)
        )
    ]
    assert "hor_dispersion_weight" not in visible_text
    assert "ver_dispersion_weight" not in visible_text

    class Event:
        ignored = False
        def ignore(self): self.ignored = True
    before = window.loco_hor_dispersion_weight.value()
    event = Event()
    window.loco_hor_dispersion_weight.clearFocus()
    window.loco_hor_dispersion_weight.wheelEvent(event)
    assert event.ignored
    assert window.loco_hor_dispersion_weight.value() == before

    window.constraint_enabled.setChecked(False)
    assert not window.constraint_quad_sigma.isEnabled()
    window.project.modified = False
    window.close()


def test_petra_machine_detection_and_coupling_routing():
    at = pytest.importorskip("at")
    lattice_path = __import__("pathlib").Path(__file__).parents[1] / "Examples/PETRAIII/data/p3_low_beta.mat"
    ring = at.load_lattice(lattice_path)
    assert len(detect_machine_elements(ring, "bpm")) > 0
    assert len(detect_machine_elements(ring, "hcor")) > 0
    assert len(detect_machine_elements(ring, "vcor")) > 0
    assert len(detect_machine_elements(ring, "quad")) > 0
    assert len(detect_machine_elements(ring, "skew")) == 16
    assert len(detect_machine_elements(ring, "cavity")) > 0
    config = LocoConfiguration()
    for name in ("hbpm_coupling", "vbpm_coupling", "hcor_coupling", "vcor_coupling"):
        setattr(config.parameters, name, True)
    mapping = config.to_backend_mapping()
    for name in ("hbpm_coupling", "vbpm_coupling", "hcor_coupling", "vcor_coupling"):
        assert name in mapping["LOCOOptions"]["fit_list"]
        assert name in mapping["FitInitConfig"]["fit_list"]


def test_petra_all_four_coupling_blocks_are_numerically_updated(tmp_path):
    import copy
    at = pytest.importorskip("at")
    from pyLOCO.config import FitInitConfig, RMConfig
    from pyLOCO.pyloco import pyloco
    from pyLOCO.response_matrix import response_matrix
    root = __import__("pathlib").Path(__file__).parents[1]
    ring = at.load_lattice(root / "Examples/PETRAIII/data/p3_low_beta.mat")
    bpms = np.asarray(detect_machine_elements(ring, "bpm")[:8], dtype=int)
    hcor = np.asarray(detect_machine_elements(ring, "hcor")[:4], dtype=int)
    vcor = np.asarray(detect_machine_elements(ring, "vcor")[:4], dtype=int)
    steps = [np.full(4, 1e-5), np.full(4, 1e-5)]
    known = {"hbpm_coupling": np.linspace(.002, .003, 8), "vbpm_coupling": np.linspace(-.0015, -.0025, 8),
             "hcor_coupling": np.linspace(.003, .004, 4), "vcor_coupling": np.linspace(-.002, -.003, 4)}
    measured = response_matrix(copy.deepcopy(ring), config=RMConfig(
        dkick=steps, bpm_ords=bpms, cm_ords=[hcor, vcor], HCMCoupling=known["hcor_coupling"],
        VCMCoupling=known["vcor_coupling"], includeDispersion=False))
    measured = np.block([[np.eye(8), np.diag(known["hbpm_coupling"])],
                         [np.diag(known["vbpm_coupling"]), np.eye(8)]]) @ measured
    fit_list = list(known)
    result = pyloco(copy.deepcopy(ring), algorithm="lm", nIter=1, nLMIter=4,
        used_bpms_ords=bpms, used_cor_ords=[hcor, vcor], quads_ords=np.array([], dtype=int),
        skew_ords=np.array([], dtype=int), CAVords=np.array([], dtype=int), nHBPM=8, nVBPM=8,
        nHorCOR=4, nVerCOR=4, quads_tilt_ind=np.array([], dtype=int), orm_measured=measured,
        weights=np.ones((16, 1)), includeDispersion=False, measured_eta_x=np.zeros(8), measured_eta_y=np.zeros(8),
        CMstep=steps, rfStep=-3000., fit_list=fit_list, remove_coupling_=False, outlier_rejection=False,
        apply_normalization=False, svd_selection_method="threshold", svd_threshold=1e-10,
        show_svd_plot=False, fit_cfg=FitInitConfig(fit_list=fit_list, CMstep=steps, individuals=True),
        output_dir=tmp_path)
    final = result[1][max(result[1])]
    assert list(result[-1]) == fit_list
    for name, expected in known.items():
        assert np.allclose(np.asarray(final[name]), expected, rtol=.01, atol=1e-5)
    assert result[5][-1] < 1e-18
    assert not (tmp_path / "jacobians" / "full").exists()


def test_gui_backend_appends_dispersion_as_response_matrix_column():
    measured = {
        "orm": np.arange(24.0).reshape(4, 6),
        "eta_x": np.array([0.1, 0.2]),
        "eta_y": np.array([-0.1, -0.2]),
        "noise_x": np.ones(2),
        "noise_y": np.ones(2),
        "dispersion_supplied": True,
    }
    indices = {
        "nHBPM": 2, "nVBPM": 2, "nHorCOR": 3, "nVerCOR": 3,
        "used_bpms_ords": np.array([1, 2]),
        "used_cor_ords": [np.array([3, 4, 5]), np.array([6, 7, 8])],
    }
    rm_cfg = SimpleNamespace(
        dkick=(1e-5, 1e-5), includeDispersion=True, rfStep=-3000.0,
        fixedpathlength=False,
    )
    fit_cfg = SimpleNamespace(individuals=True)
    fixed = SimpleNamespace(rfstep=-3000.0, Frequency=5e8)
    kwargs = _build_pyloco_kwargs(
        ring=None, options={"includeDispersion": True, "hor_dispersion_weight": 2.5, "ver_dispersion_weight": 3.5}, rm_cfg=rm_cfg,
        fit_cfg=fit_cfg, constraint_cfg=None, fixed_parameters=fixed,
        measured=measured, indices=indices,
    )
    assert kwargs["orm_measured"].shape == (4, 7)
    assert kwargs["orm_measured"][:, -1].tolist() == [0.1, 0.2, -0.1, -0.2]
    assert kwargs["rfStep"] == -3000.0
    assert kwargs["hor_dispersion_weight"] == 2.5
    assert kwargs["ver_dispersion_weight"] == 3.5


def test_gui_backend_refuses_unhonored_acquisition_conventions():
    measured = {"orm": np.zeros((4, 2)), "eta_x": np.zeros(2), "eta_y": np.zeros(2),
                "noise_x": np.ones(2), "noise_y": np.ones(2), "dispersion_supplied": False}
    indices = {"nHBPM": 2, "nVBPM": 2, "nHorCOR": 1, "nVerCOR": 1,
               "used_bpms_ords": np.array([1, 2]), "used_cor_ords": [np.array([3]), np.array([4])]}
    base = dict(dkick=(1e-5, 1e-5), includeDispersion=False, rfStep=-3000., fixedpathlength=False)
    common = dict(ring=None, options={}, fit_cfg=SimpleNamespace(individuals=True), constraint_cfg=None,
                  fixed_parameters=SimpleNamespace(rfstep=-3000., Frequency=5e8), measured=measured, indices=indices)
    with pytest.raises(ValueError, match="Tracking mode"):
        _build_pyloco_kwargs(rm_cfg=SimpleNamespace(**base, calculator="Tracking", bidirectional=True), **common)
    with pytest.raises(ValueError, match="one-sided"):
        _build_pyloco_kwargs(rm_cfg=SimpleNamespace(**base, calculator="Linear", bidirectional=False), **common)


def test_petra_gui_dispersion_payload_matches_canonical_workflow_after_exclusions():
    root = __import__("pathlib").Path(__file__).parents[1]
    measured = _load_measurements({
        "orm": str(root / "Examples/PETRAIII/data/measured_orm_loco.h5"),
        "dispersion": str(root / "Examples/PETRAIII/data/measured_dispersion_loco.h5"),
        "bpm_noise": str(root / "Examples/PETRAIII/data/measured_BPM_noise_loco.h5"),
    })
    # Canonical Examples/PETRAIII/petra_workflow.py ordering.
    canonical = np.hstack((measured["orm"], np.concatenate((measured["eta_x"], measured["eta_y"]))[:, None]))
    assert np.array_equal(_assemble_measured_response(measured, True), canonical)
    assert canonical.shape == (492, 414)
    assert np.array_equal(canonical[:246, -1], measured["eta_x"])
    assert np.array_equal(canonical[246:, -1], measured["eta_y"])


def test_dispersion_metadata_applies_rf_step_and_measurement_direction(app, tmp_path):
    path = tmp_path / "dispersion.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("measured_eta_x", data=[1.0, 2.0])
        handle.create_dataset("measured_eta_y", data=[3.0, 4.0])
        handle.attrs["rf_step_hz"] = -2750.0; handle.attrs["bidirectional"] = False
    metadata = inspect_measurement_metadata(path, "dispersion")
    window = MainWindow(); window._apply_measurement_metadata("dispersion", metadata)
    assert window.rm_rf_step.value() == pytest.approx(-2750.0)
    assert not window.rm_bidirectional.isChecked()
    assert "measured_eta_x" in window.measurement_metadata_label.text()
    window.project.modified = False
    window.close()


def test_interactive_exclusion_dialog_maps_position_to_lattice_ordinal(app):
    window = MainWindow()
    dialog = ExclusionSelectionDialog(window, "Exclude", [10, 20, 30], [1])
    assert dialog.table.item(1, 1).text() == "1"
    assert dialog.table.item(1, 2).text() == "20"
    assert dialog._selected() == [1]
    dialog.table.item(2, 0).setCheckState(Qt.Checked)
    assert dialog._selected() == [1, 2]
    dialog.close(); window.close()


def test_persisted_optics_and_jacobian_are_exactly_reloadable(tmp_path):
    at = pytest.importorskip("at")
    root = __import__("pathlib").Path(__file__).parents[1]
    lattice_path = root / "Examples/PETRAIII/data/p3_low_beta.mat"
    ring = at.load_lattice(lattice_path); bpms = np.asarray(at.get_refpts(ring, at.elements.Monitor), dtype=int)[:5]
    optics_path = _save_optics(tmp_path, lattice_path, ring, {"used_bpms_ords": bpms})
    assert optics_path and optics_path.exists()
    matrix = np.arange(24.0).reshape(8, 3)
    request = SimpleNamespace(backend_mapping={"RMConfig": {"calculator": "Linear"}, "LOCOOptions": {"includeDispersion": True, "save_jacobians": True}, "FixedParameters": {"dk": 1e-4}})
    files = _save_jacobian(tmp_path, {"matrix": matrix, "iteration": 2}, {"quads": slice(0, 3)}, request,
                           {"used_bpms_ords": bpms, "used_cor_ords": [np.array([1]), np.array([2])]})
    assert len(files) == 2
    loader = ResultsLoader(tmp_path)
    assert np.array_equal(loader.jacobian, matrix)
    assert loader.jacobian_metadata["shape"] == [8, 3]
    assert loader.beta_beating is not None
    assert np.allclose(loader.beta_beating["beating"], 0.0)

    disabled = SimpleNamespace(backend_mapping={"LOCOOptions": {"save_jacobians": False}})
    disabled_dir = tmp_path / "disabled"; disabled_dir.mkdir()
    assert _save_jacobian(disabled_dir, {"matrix": matrix, "iteration": 2}, {}, disabled, {}) == []
    assert not (disabled_dir / "jacobian.h5").exists()


def _synthetic_results(path):
    path.mkdir()
    (path / "summary.json").write_text(json.dumps({"initial_chi2": 10.0, "chi2_history": [5.0, 2.0],
                                                   "runtime_seconds": 1.25,
                                                   "blocks": {"hbpm_gain": {"start": 0, "stop": 2}}}))
    (path / "run_request.json").write_text(json.dumps({"project_name": "test", "lattice_path": "missing.mat",
        "measurements": {}, "backend_mapping": {"LOCOOptions": {"fit_list": ["hbpm_gain"]},
        "MachineElements": {}, "FitInitConfig": {}, "ConstraintConfig": {}}}))
    np.savez_compressed(path / "loco_results.npz", fit_results=np.array([[1.0, 1.0], [1.1, .9]]), chi2_history=np.array([5., 2.]))


def test_completed_gui_run_is_recorded_saved_and_restored(app, tmp_path, monkeypatch):
    result_dir = tmp_path / "results"; _synthetic_results(result_dir)
    project_path = tmp_path / "completed.pyloco.json"
    monkeypatch.setattr("pyLOCO.gui.main_window.QMessageBox.information", lambda *args: None)

    window = MainWindow()
    window.project.path = str(project_path)
    result = SimpleNamespace(
        results_dir=str(result_dir), elapsed_seconds=3.5,
        output_files=[], chi2_history=[5.0, 2.0],
    )
    window._on_loco_finished(result)
    assert window.project.completed_run.results_dir == str(result_dir)
    assert window.project.completed_run.status == "completed"
    assert window.project.modified
    window.save_project()

    saved = ProjectMetadata.load(project_path)
    assert saved.completed_run.results_dir == str(result_dir)
    reopened = MainWindow(); reopened.open_project(project_path); app.processEvents()
    assert reopened.results_workspace.loader is not None
    assert reopened.results_workspace.loader.result_dir == result_dir
    assert list(reopened.results_workspace.loader.chi2_history) == [5.0, 2.0]
    reopened.close(); window.close()


def test_completed_run_reload_restores_chi2_and_summary(app, tmp_path):
    result_dir = tmp_path / "results"; _synthetic_results(result_dir)
    project = ProjectMetadata(completed_run=CompletedRunReference(str(result_dir), 1.25, "completed"))
    restored = ProjectMetadata.load(project.save(tmp_path / "run.pyloco.json"))
    workspace = ResultsWorkspace(); workspace.load_results(restored.completed_run.results_dir, runtime=restored.completed_run.elapsed_seconds)
    assert workspace.loader.chi2_history == [5.0, 2.0]
    assert workspace.loader.final_chi2 == 2.0
    assert not workspace.run_progress.isVisible()


def test_complete_nondefault_gui_state_survives_save_close_and_reload(app, tmp_path):
    project = ProjectMetadata(name="nondefault", mode="Advanced")
    cfg = project.loco_config
    cfg.machine_elements.bpm_ords = [10, 20, 30]
    cfg.machine_elements.horizontal_corrector_ords = [40, 41]
    cfg.machine_elements.vertical_corrector_ords = [50, 51]
    cfg.machine_elements.normal_quadrupole_ords = [60, 61]
    cfg.machine_elements.skew_quadrupole_ords = [70]
    cfg.machine_elements.cavity_ords = [80]
    cfg.bad_bpm_positions = [1]; cfg.excluded_horizontal_corrector_positions = [0]; cfg.excluded_vertical_corrector_positions = [1]
    cfg.element_selection_state = {"bpm_ords": {"method": "manual", "indices": [10, 20, 30]}}
    cfg.rejection.includeDispersion = True; cfg.rejection.hor_dispersion_weight = 2.25; cfg.rejection.ver_dispersion_weight = 3.5
    cfg.constraints.enable = True; cfg.constraints.quad_sigma = 0.004; cfg.constraints.skew_sigma = 0.006
    cfg.solver.algorithm = "lm"; cfg.solver.nIter = 3; cfg.solver.nLMIter = 7; cfg.solver.Starting_Lambda = 0.025
    cfg.svd.svd_threshold = 2e-8; cfg.parameters.quads = True; cfg.parameters.skew_quads = True; cfg.parameters.hbpm_coupling = True
    cfg.response_matrix.dkick_h = 1.2e-5; cfg.response_matrix.dkick_v = 2.3e-5; cfg.response_matrix.bidirectional = False; cfg.response_matrix.rfStep = -2450.0
    measurement = tmp_path / "orm.h5"
    with h5py.File(measurement, "w") as handle: handle.create_dataset("response_matrix", data=np.zeros((6, 4)))
    project.measurements["orm"] = ImportedDataset("orm", str(measurement), "h5", options={"dkick_h": 1.2e-5, "bidirectional": False})
    first = MainWindow(); first.project = project; first._load_config_to_widgets()
    before = first._collect_loco_configuration().to_backend_mapping(); saved = project.save(tmp_path / "full.pyloco.json"); first.close()
    second = MainWindow(); second.project = ProjectMetadata.load(saved); second._load_config_to_widgets()
    after = second._collect_loco_configuration().to_backend_mapping()
    assert after == before
    assert second.project.measurements["orm"].options == {"dkick_h": 1.2e-5, "bidirectional": False}
    assert second.bad_bpm_positions_edit.text() == "1"
    assert second.hcor_exclusions_edit.text() == "0"
    second.close()


def test_element_selection_provenance_round_trips_without_changing_authoritative_ordinals(tmp_path):
    project = ProjectMetadata(name="selection-provenance")
    ordinals = [7, 11, 19]
    project.loco_config.machine_elements.normal_quadrupole_ords = ordinals
    project.loco_config.element_selection_state["normal_quadrupole_ords"] = {
        "version": 1,
        "method": "at_wildcard",
        "patterns": "QD3[AE]*\nQF4[ABDE]*\nQD5[BD]*",
        "generated_ords": [7, 11, 17, 19],
        "resolved_ords": ordinals,
        "manual_additions": [],
        "manual_removals": [17],
    }
    restored = ProjectMetadata.load(project.save(tmp_path / "selection.pyloco.json"))
    assert restored.loco_config.machine_elements.normal_quadrupole_ords == ordinals
    assert restored.loco_config.element_selection_state["normal_quadrupole_ords"] == (
        project.loco_config.element_selection_state["normal_quadrupole_ords"]
    )


def test_save_project_snapshots_live_widgets_and_overwrites_loaded_file(app, tmp_path):
    target = tmp_path / "live-state.pyloco.json"
    first = MainWindow()
    first.project.path = str(target)
    first.project.base_directory = str(tmp_path)
    first.project.loco_config.machine_elements.bpm_ords = [4, 8, 12]
    first.project.loco_config.element_selection_state = {
        "bpm_ords": {"method": "manual", "indices": [4, 8, 12]}
    }
    first.project.measurements["orm"] = ImportedDataset(
        "orm", str(tmp_path / "orm.h5"), "h5",
        options={"dataset": "response_matrix", "bidirectional": False},
    )
    first.dashboard_name.setText("Saved from widgets")
    first.dashboard_description.setText("Portable PETRA test project")
    first.rm_calculator.setCurrentIndex(first.rm_calculator.findData("Numerical"))
    first.rm_dkick_h.setValue(1.25e-5)
    first.rm_dkick_v.setValue(2.5e-5)
    first.rm_rf_step.setValue(-2750.0)
    first.rm_dispersion.setChecked(False)
    first.loco_hor_dispersion_weight.setValue(5.0)
    first.loco_ver_dispersion_weight.setValue(2.5)
    first.solver_algorithm.setCurrentIndex(first.solver_algorithm.findData("gn"))
    first.solver_n_iter.setValue(4)
    first.solver_lm_iter.setValue(9)
    first.solver_lambda.setValue(0.125)
    first.solver_max_lambda.setValue(21.0)
    first.bad_bpm_positions_edit.setText("1, 2")
    first.output_directory_edit.setText(str(tmp_path / "output"))
    first.run_name_edit.setText("student-run")
    first.save_jacobian_check.setChecked(True)

    first.save_project()
    assert target.exists()
    assert first.project.path == str(target.resolve())
    first.close()

    second = MainWindow()
    second.open_project(target)
    assert second.dashboard_name.text() == "Saved from widgets"
    assert second.dashboard_description.text() == "Portable PETRA test project"
    assert second.rm_calculator.currentData() == "Numerical"
    assert second.rm_dkick_h.value() == pytest.approx(1.25e-5)
    assert second.rm_dkick_v.value() == pytest.approx(2.5e-5)
    assert second.rm_rf_step.value() == pytest.approx(-2750.0)
    assert not second.rm_dispersion.isChecked()
    assert second.loco_hor_dispersion_weight.value() == 5.0
    assert second.loco_ver_dispersion_weight.value() == 2.5
    assert second.solver_algorithm.currentData() == "gn"
    assert second.solver_lm_iter.value() == 9
    assert second.solver_lambda.value() == pytest.approx(0.125)
    assert second.bad_bpm_positions_edit.text() == "1, 2"
    assert second.run_name_edit.text() == "student-run"
    assert second.save_jacobian_check.isChecked()
    assert second.project.loco_config.machine_elements.bpm_ords == [4, 8, 12]
    assert second.project.measurements["orm"].options["dataset"] == "response_matrix"

    # A normal Save overwrites the loaded file with the newest live widget state.
    second.dashboard_name.setText("Newest state")
    second.solver_n_iter.setValue(6)
    second.save_project()
    second.close()
    third = ProjectMetadata.load(target)
    assert third.name == "Newest state"
    assert third.loco_config.solver.nIter == 6


def test_new_incomplete_project_saves_case_c_measurement_bindings_and_reopens(app, tmp_path, monkeypatch):
    root = __import__("pathlib").Path(__file__).parents[1]
    data = root / "Examples/EBS/multistage_fit/data"
    sources = {
        "bpm_noise": data / "measured_BPM_noise_loco_20260126_212023.h5",
        "dispersion": data / "measured_dispersion_loco_20260126_221103.h5",
        "orm": data / "ORM_for_pyLOCO_20260126_221131.h5",
    }
    window = MainWindow()
    for role, source in sources.items():
        window.measurement_role.setCurrentText(role)
        monkeypatch.setattr(
            "pyLOCO.gui.main_window.QFileDialog.getOpenFileName",
            lambda *args, source=source, **kwargs: (str(source), ""),
        )
        window.import_measurement()
    window.bad_bpm_positions_edit.setText("1, 27")
    window.hcor_exclusions_edit.setText("2")
    window.vcor_exclusions_edit.setText("3")
    target = tmp_path / "case-c-incomplete.pyloco.json"
    observed = {}

    def choose_project(*args, **kwargs):
        observed["options"] = kwargs.get("options")
        return str(target), ""

    monkeypatch.setattr("pyLOCO.gui.main_window.QFileDialog.getSaveFileName", choose_project)
    assert window.save_project() is True
    assert target.exists()
    assert "FIT configuration incomplete" in window._last_save_status
    assert observed["options"] & QFileDialog.DontUseNativeDialog
    window.close()

    restored = ProjectMetadata.load(target)
    assert list(restored.measurements) == ["bpm_noise", "dispersion", "orm"]
    assert {role: restored.measurements[role].path for role in sources} == {
        role: str(source.resolve()) for role, source in sources.items()
    }
    assert restored.loco_config.bad_bpm_positions == [1, 27]
    assert restored.loco_config.excluded_horizontal_corrector_positions == [2]
    assert restored.loco_config.excluded_vertical_corrector_positions == [3]
    payload = json.loads(target.read_text())
    assert all(isinstance(item["path"], str) for item in payload["measurements"].values())
    assert target.stat().st_size < 2_000_000  # references and metadata, never HDF5 arrays


def test_loading_case_c_recipe_preserves_measurements_and_resaves(app, tmp_path, monkeypatch):
    root = __import__("pathlib").Path(__file__).parents[1]
    historical = ProjectMetadata.load(root / "Examples/EBS/multistage_fit/EBS_case_C_multistage.pyloco.json")
    target = historical.save(tmp_path / "recipe-preservation.pyloco.json")
    window = MainWindow(); window.open_project(target)
    before = [
        (role, dataset.path, dataset.file_type, dataset.size_bytes, dict(dataset.options))
        for role, dataset in window.project.measurements.items()
    ]
    recipe = root / "Examples/EBS/multistage_fit/recipe.json"
    monkeypatch.setattr(
        "pyLOCO.gui.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(recipe), ""),
    )
    monkeypatch.setattr(window, "_choose_recipe_machine_policy", lambda: "recipe")
    window.load_fit_recipe()
    after = [
        (role, dataset.path, dataset.file_type, dataset.size_bytes, dict(dataset.options))
        for role, dataset in window.project.measurements.items()
    ]
    assert after == before
    assert window.project.modified
    assert window.save_project() is True
    window.close()
    reopened = ProjectMetadata.load(target)
    assert [(role, dataset.path) for role, dataset in reopened.measurements.items()] == [
        (role, path) for role, path, _kind, _size, _options in before
    ]


def test_missing_measurement_reference_is_explained_and_can_be_relocated(app, tmp_path, monkeypatch):
    root = __import__("pathlib").Path(__file__).parents[1]
    replacement = root / "Examples/EBS/multistage_fit/data/ORM_for_pyLOCO_20260126_221131.h5"
    missing = tmp_path / "moved-orm.h5"
    project = ProjectMetadata(name="missing measurement")
    project.measurements["orm"] = ImportedDataset(
        "orm", str(missing), "h5", 123, {"operator_note": "retain this project metadata"}
    )
    target = project.save(tmp_path / "missing.pyloco.json")
    window = MainWindow(); window.open_project(target)
    assert "Missing measurement file reference" in window.measurement_file_status.text()
    assert "moved-orm.h5" in window.measurement_file_status.text()
    assert window.locate_missing_measurement_button.isEnabled()
    monkeypatch.setattr(
        "pyLOCO.gui.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(replacement), ""),
    )
    window.locate_missing_measurement()
    assert window.project.measurements["orm"].path == str(replacement.resolve())
    assert window.project.measurements["orm"].options["operator_note"] == "retain this project metadata"
    assert "All imported measurement files are available" in window.measurement_file_status.text()
    assert window.project.modified
    window.project.modified = False
    window.close()


def test_project_save_cancel_and_failure_are_not_silent(app, tmp_path, monkeypatch):
    window = MainWindow()
    monkeypatch.setattr(
        "pyLOCO.gui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: ("", ""),
    )
    assert window.save_project() is False
    assert window._last_save_status == "Project not saved — no filename selected."

    messages = []
    monkeypatch.setattr(
        "pyLOCO.gui.main_window.QMessageBox.warning",
        lambda *args: messages.append(args[2]),
    )
    monkeypatch.setattr(
        ProjectMetadata,
        "save",
        lambda *args: (_ for _ in ()).throw(OSError("disk unavailable")),
    )
    assert window._save_project_to(tmp_path / "cannot-save.pyloco.json") is False
    assert messages and "disk unavailable" in messages[-1]
    assert window._last_save_status.startswith("Project could not be saved")
    window.project.modified = False
    window.close()


def test_save_as_creates_separate_copy_and_project_status_shows_filename(app, tmp_path, monkeypatch):
    original = ProjectMetadata(name="Original project").save(tmp_path / "original.pyloco.json")
    original_payload = original.read_bytes()
    window = MainWindow(); window.open_project(original)
    assert "original.pyloco.json" in window._project_label.text()
    assert "Saved" in window._project_label.text()
    window.dashboard_description.setText("Changed only in the copy")
    window._update_project_description()
    assert "Unsaved changes" in window._project_label.text()
    copy = tmp_path / "separate-copy.pyloco.json"
    monkeypatch.setattr(
        "pyLOCO.gui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(copy), ""),
    )
    assert window.save_project_as() is True
    assert original.read_bytes() == original_payload
    assert copy.exists() and window.project.path == str(copy.resolve())
    assert "separate-copy.pyloco.json" in window._project_label.text()
    assert "Saved" in window._project_label.text()
    window.close()


def test_case_c_machine_configuration_recipe_preservation_workflow(app, tmp_path, monkeypatch):
    from copy import deepcopy
    root = __import__("pathlib").Path(__file__).parents[1]
    case_c_path = root / "Examples/EBS/multistage_fit/EBS_case_C_multistage.pyloco.json"
    recipe_path = root / "Examples/EBS/multistage_fit/recipe.json"
    historical = ProjectMetadata.load(case_c_path)

    source_window = MainWindow()
    source_window.project.lattice = deepcopy(historical.lattice)
    machine_payload = source_window._machine_configuration_snapshot(historical.loco_config)
    machine_path = tmp_path / "case-c.pyloco-machine.json"
    machine_path.write_text(json.dumps(machine_payload), encoding="utf-8")
    source_window.close()

    partial = ProjectMetadata(
        name="new Case C measurements",
        lattice=deepcopy(historical.lattice),
        measurements=deepcopy(historical.measurements),
    )
    partial.loco_config.solver.nIter = 77
    partial_path = partial.save(tmp_path / "partial.pyloco.json")
    window = MainWindow(); window.open_project(partial_path)
    measurement_bindings = [(role, dataset.path) for role, dataset in window.project.measurements.items()]
    dialogs = iter((str(machine_path), str(recipe_path)))
    monkeypatch.setattr(
        "pyLOCO.gui.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (next(dialogs), ""),
    )
    monkeypatch.setattr(
        "pyLOCO.gui.main_window.QMessageBox.question",
        lambda *args, **kwargs: QMessageBox.Yes,
    )
    assert window.load_machine_configuration() is True
    expected_elements = __import__("dataclasses").asdict(historical.loco_config.machine_elements)
    assert __import__("dataclasses").asdict(window.project.loco_config.machine_elements) == expected_elements
    assert window.project.loco_config.bad_bpm_positions == historical.loco_config.bad_bpm_positions
    assert window.project.loco_config.element_selection_state == historical.loco_config.element_selection_state
    assert window.project.loco_config.solver.nIter == 77
    monkeypatch.setattr(window, "_choose_recipe_machine_policy", lambda: "current")
    window.load_fit_recipe()
    assert window.project.recipe_machine_policy == "current"
    assert len(window.fit_recipe.stages) == 4
    assert [(role, dataset.path) for role, dataset in window.project.measurements.items()] == measurement_bindings
    for row in range(4):
        window.fit_workflow_table.selectRow(row); app.processEvents()
        assert __import__("dataclasses").asdict(window.project.loco_config.machine_elements) == expected_elements
        assert window.project.loco_config.bad_bpm_positions == historical.loco_config.bad_bpm_positions
    assert window.save_project() is True
    window.close()

    reopened = MainWindow(); reopened.open_project(partial_path)
    assert reopened.project.recipe_machine_policy == "current"
    assert [(role, dataset.path) for role, dataset in reopened.project.measurements.items()] == measurement_bindings
    for row in range(4):
        reopened.fit_workflow_table.selectRow(row); app.processEvents()
        assert __import__("dataclasses").asdict(reopened.project.loco_config.machine_elements) == expected_elements
    reopened.close()


def test_machine_configuration_rejects_lattice_and_ordinal_errors_transactionally(app, tmp_path):
    from copy import deepcopy
    root = __import__("pathlib").Path(__file__).parents[1]
    historical = ProjectMetadata.load(root / "Examples/EBS/multistage_fit/EBS_case_C_multistage.pyloco.json")
    window = MainWindow(); window.project.lattice = deepcopy(historical.lattice)
    before = __import__("dataclasses").asdict(window.project.loco_config.machine_elements)
    payload = window._machine_configuration_snapshot(historical.loco_config)
    for label, mutate, match in (
        ("checksum", lambda value: value["reference_lattice"].update(sha256="0" * 64), "checksum"),
        ("count", lambda value: value["reference_lattice"].update(element_count=4433), "element count"),
        ("ordinal", lambda value: value["machine_elements"]["bpm_ords"].__setitem__(0, 999999), "lattice range"),
    ):
        candidate = deepcopy(payload); mutate(candidate)
        path = tmp_path / f"invalid-{label}.json"; path.write_text(json.dumps(candidate))
        with pytest.raises(ValueError, match=match):
            window._validated_machine_configuration(path)
        assert __import__("dataclasses").asdict(window.project.loco_config.machine_elements) == before
    window.close()


def test_preserved_machine_configuration_rejects_recipe_corrector_count_mismatch(app):
    from copy import deepcopy
    from pyLOCO.gui.fit_workflow import FitRecipe
    root = __import__("pathlib").Path(__file__).parents[1]
    historical = ProjectMetadata.load(root / "Examples/EBS/multistage_fit/EBS_case_C_multistage.pyloco.json")
    window = MainWindow()
    window.project.lattice = deepcopy(historical.lattice)
    window.project.measurements = deepcopy(historical.measurements)
    candidate = FitRecipe.load(root / "Examples/EBS/multistage_fit/recipe.json")
    config = deepcopy(historical.loco_config)
    config.machine_elements.horizontal_corrector_ords = config.machine_elements.horizontal_corrector_ords[:-1]
    override = window._machine_override_from_config(config)
    window._apply_machine_override_to_recipe(candidate, override)
    errors = window._preflight_imported_recipe(candidate)
    assert any("selects 31 horizontal correctors" in error for error in errors)
    window.close()


def _case_c_project_with_copied_measurements(tmp_path):
    from copy import deepcopy
    import shutil
    root = __import__("pathlib").Path(__file__).parents[1]
    historical = ProjectMetadata.load(root / "Examples/EBS/multistage_fit/EBS_case_C_multistage.pyloco.json")
    active = ProjectMetadata(name="new measurement handoff")
    active.measurements = deepcopy(historical.measurements)
    for role, dataset in active.measurements.items():
        source = __import__("pathlib").Path(dataset.path)
        target = tmp_path / "new-measurements" / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        dataset.path = str(target)
    active.measurement_session = {
        "session_id": "new-compatible-case-c-session",
        "manifest": str(tmp_path / "new-measurements/session.pyloco-session.json"),
        "measurement_files": {role: dataset.path for role, dataset in active.measurements.items()},
    }
    return historical, active


def test_open_project_can_keep_compatible_active_measurements_and_invalidate_runs(app, tmp_path, monkeypatch):
    historical, active = _case_c_project_with_copied_measurements(tmp_path)
    candidate_path = historical.save(tmp_path / "historical-project.pyloco.json")
    window = MainWindow(); window.project = active; window.project.modified = False; window._refresh_ui("handoff")
    active_paths = {role: dataset.path for role, dataset in active.measurements.items()}
    monkeypatch.setattr(window, "_choose_open_project_measurement_policy", lambda candidate: "keep")
    window.open_project(candidate_path)
    assert window._workspace.tabText(window._workspace.currentIndex()) == "Machine Components"
    assert "Project configuration loaded" in window.statusBar().currentMessage()
    assert "Current measurements preserved" in window.statusBar().currentMessage()
    assert {role: dataset.path for role, dataset in window.project.measurements.items()} == active_paths
    assert window.project.measurement_session["session_id"] == "new-compatible-case-c-session"
    assert window.project.completed_run.results_dir == ""
    assert window.project.fit_run_session == {}
    assert window.project.modified is True
    assert "Unsaved changes" in window._project_label.text()
    assert len(window.fit_recipe.stages) == 4
    assert [stage.name for stage in window.fit_recipe.stages] == [
        "Calibration", "BPM coupling", "Normal optics", "Skew and coupling"
    ]
    assert window.fit_workflow_table.rowCount() == 4
    assert [window.fit_workflow_table.item(row, 1).text() for row in range(4)] == [
        "Calibration", "BPM coupling", "Normal optics", "Skew and coupling"
    ]
    assert __import__("dataclasses").asdict(window.project.loco_config.machine_elements) == __import__("dataclasses").asdict(
        historical.loco_config.machine_elements
    )
    assert window.project.loco_config.bad_bpm_positions == historical.loco_config.bad_bpm_positions
    assert window.project.loco_config.excluded_horizontal_corrector_positions == historical.loco_config.excluded_horizontal_corrector_positions
    assert window.project.loco_config.excluded_vertical_corrector_positions == historical.loco_config.excluded_vertical_corrector_positions
    expected_elements = __import__("dataclasses").asdict(historical.loco_config.machine_elements)
    for row in range(4):
        window.fit_workflow_table.selectRow(row); app.processEvents()
        assert __import__("dataclasses").asdict(window.project.loco_config.machine_elements) == expected_elements
        assert window.project.loco_config.bad_bpm_positions == historical.loco_config.bad_bpm_positions
        assert window.project.loco_config.excluded_horizontal_corrector_positions == historical.loco_config.excluded_horizontal_corrector_positions
        assert window.project.loco_config.excluded_vertical_corrector_positions == historical.loco_config.excluded_vertical_corrector_positions
    assert "new-compatible-case-c-session" in window.active_measurement_session_label.text()
    assert all(path not in window.active_measurement_session_label.text() for path in active_paths.values())
    assert window.measurement_list.columnCount() == 6
    assert {window.measurement_list.item(row, 0).text() for row in range(3)} == {
        "Bpm Noise", "Dispersion", "Orm"
    }
    assert all(window.measurement_list.item(row, 4).text() == "Loaded" for row in range(3))
    assert all(window.measurement_list.item(row, 2).text() != "" for row in range(3))
    assert all(window.measurement_list.item(row, 3).text() != "" for row in range(3))
    detail_buttons = [window.measurement_list.cellWidget(row, 5).findChild(QToolButton) for row in range(3)]
    assert all(button is not None and button.text() == "Details…" for button in detail_buttons)
    assert all(button.height() <= window.measurement_list.rowHeight(row) for row, button in enumerate(detail_buttons))
    assert all(active_paths[role] in window.measurement_list.item(row, 1).toolTip()
               for row, role in enumerate(sorted(active_paths)))
    detail_messages = []
    monkeypatch.setattr(
        "pyLOCO.gui.main_window.QMessageBox.information",
        lambda *args: detail_messages.append(args[2]),
    )
    detail_buttons[0].click()
    assert detail_messages
    assert "Directory:" in detail_messages[-1]
    assert "Datasets and dimensions:" in detail_messages[-1]
    assert active_paths[sorted(active_paths)[0]] in detail_messages[-1]
    copy_path = tmp_path / "new-measurement-fit.pyloco.json"
    monkeypatch.setattr(
        "pyLOCO.gui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(copy_path), ""),
    )
    assert window.save_project_as() is True
    window.close()
    reopened = ProjectMetadata.load(copy_path)
    assert {role: dataset.path for role, dataset in reopened.measurements.items()} == active_paths
    assert reopened.measurement_session["session_id"] == "new-compatible-case-c-session"


def test_keep_current_measurements_preserves_revalidated_project_mapping_and_requires_validation(
    app, tmp_path, monkeypatch
):
    historical, active = _case_c_project_with_copied_measurements(tmp_path)
    saved_mapping = {
        "schema": "pyloco.measurement_device_mapping",
        "schema_version": 1,
        "inventory": {"sha256": "active-inventory"},
        "mappings": [{"role": "bpm", "measured_name": "BPM1", "status": "mapped"}],
    }
    historical.measurement_device_mapping = deepcopy(saved_mapping)
    historical.compatibility_validation = {"required": True, "status": "validated", "issues": []}
    candidate_path = historical.save(tmp_path / "mapped-project.pyloco.json")
    active.measurement_device_mapping = {}
    window = MainWindow(); window.project = active; window.project.modified = False; window._refresh_ui("handoff")

    monkeypatch.setattr(window, "_choose_open_project_measurement_policy", lambda candidate: "keep")

    def reconcile(candidate):
        assert candidate.measurement_device_mapping == saved_mapping
        candidate.measurement_device_mapping = {
            **deepcopy(saved_mapping),
            "mapping_provenance": "revalidated against retained measurements",
        }
        return []

    monkeypatch.setattr(window, "_reconcile_retained_measurement_mapping", reconcile)
    monkeypatch.setattr(window, "_validate_retained_measurements", lambda candidate: [])
    window.open_project(candidate_path)

    assert window.project.measurement_device_mapping["mapping_provenance"] == (
        "revalidated against retained measurements"
    )
    assert window.project.compatibility_validation["status"] == "validation_required"
    assert window.project.compatibility_validation["required"] is True
    assert window.run_loco_action.isEnabled() is False

    saved_copy = tmp_path / "retained-measurements-save-as.pyloco.json"
    assert window._save_project_to(saved_copy) is True
    restored = ProjectMetadata.load(saved_copy)
    assert restored.measurement_device_mapping == window.project.measurement_device_mapping
    assert restored.compatibility_validation["status"] == "validation_required"
    window.project.modified = False
    window.close()


def test_reconcile_retained_mapping_invalidates_stale_mapping_without_guessing(app, tmp_path, monkeypatch):
    historical, active = _case_c_project_with_copied_measurements(tmp_path)
    historical.measurements = deepcopy(active.measurements)
    historical.measurement_device_mapping = {
        "schema": "pyloco.measurement_device_mapping",
        "schema_version": 1,
        "inventory": {"sha256": "stale"},
        "mappings": [{"role": "bpm", "measured_name": "old", "status": "mapped"}],
    }
    lattice_path = tmp_path / "ring.mat"
    lattice_path.write_bytes(b"ring")
    historical.lattice.path = str(lattice_path)
    historical.lattice.mat_variable = "RING"
    payload = {
        "errors": [],
        "files": {"orm": {
            "bpm_names": ["BPM1"],
            "horizontal_corrector_names": ["HC1-B1L"],
            "vertical_corrector_names": ["VC1-A1L"],
        }},
    }
    monkeypatch.setattr("pyLOCO.gui.main_window.inspect_measurement_files", lambda *args, **kwargs: payload)
    monkeypatch.setattr("pyLOCO.gui.main_window.load_project_lattice", lambda *args, **kwargs: [object()])
    calls = []

    def reject(*args, **kwargs):
        calls.append((args, kwargs))
        raise ValueError("Mapping device inventory differs from the current measurement inventory")

    monkeypatch.setattr("pyLOCO.gui.main_window.validate_measurement_mapping", reject)
    window = MainWindow()
    errors = window._reconcile_retained_measurement_mapping(historical)
    assert len(calls) == 1
    assert historical.measurement_device_mapping == {}
    assert errors and "invalidated" in errors[0]
    assert "inventory differs" in errors[0]
    window.close()


def test_reconcile_retained_mapping_requires_mapping_for_identified_inventory(app, tmp_path, monkeypatch):
    historical, active = _case_c_project_with_copied_measurements(tmp_path)
    historical.measurements = deepcopy(active.measurements)
    historical.measurement_device_mapping = {}
    historical.compatibility_validation = {"required": True, "status": "validated", "issues": []}
    payload = {
        "errors": [],
        "files": {"orm": {
            "bpm_names": ["BPM1"],
            "horizontal_corrector_names": ["HC1-B1L"],
            "vertical_corrector_names": ["VC1-A1L"],
        }},
    }
    monkeypatch.setattr("pyLOCO.gui.main_window.inspect_measurement_files", lambda *args, **kwargs: payload)
    window = MainWindow()
    errors = window._reconcile_retained_measurement_mapping(historical)
    assert errors and "mapping is missing" in errors[0]
    assert historical.measurement_device_mapping == {}
    window.close()


def test_open_project_unsaved_decision_is_optional_and_precedes_measurement_choice(app, tmp_path, monkeypatch):
    historical, active = _case_c_project_with_copied_measurements(tmp_path)
    candidate_path = historical.save(tmp_path / "historical-project.pyloco.json")
    window = MainWindow(); window.project = active; window.project.modified = True; window._refresh_ui("handoff")
    before = window._measurement_binding_signature(window.project)
    decisions = []
    monkeypatch.setattr(window, "_confirm_open_with_unsaved_changes", lambda: decisions.append("continue") or True)
    monkeypatch.setattr(window, "_choose_open_project_measurement_policy", lambda candidate: decisions.append("keep") or "keep")
    monkeypatch.setattr(window, "save_project", lambda: (_ for _ in ()).throw(AssertionError("saving is optional")))
    window.open_project(candidate_path)
    assert decisions == ["continue", "keep"]
    assert window._measurement_binding_signature(window.project) == before
    assert [stage.name for stage in window.fit_recipe.stages] == [
        "Calibration", "BPM coupling", "Normal optics", "Skew and coupling"
    ]
    window.project.modified = False
    window.close()


def test_open_project_unsaved_cancel_leaves_everything_unchanged(app, tmp_path, monkeypatch):
    historical, active = _case_c_project_with_copied_measurements(tmp_path)
    candidate_path = historical.save(tmp_path / "historical-project.pyloco.json")
    window = MainWindow(); window.project = active; window.project.modified = True; window._refresh_ui("handoff")
    before = window.project.to_dict()
    monkeypatch.setattr(window, "_confirm_open_with_unsaved_changes", lambda: False)
    monkeypatch.setattr(
        window, "_choose_open_project_measurement_policy",
        lambda candidate: (_ for _ in ()).throw(AssertionError("measurement choice must not appear after cancellation")),
    )
    window.open_project(candidate_path)
    assert window.project.to_dict() == before
    window.project.modified = False
    window.close()


def test_manual_case_c_import_then_open_four_stage_project_keeps_measurements_and_loads_configuration(app, monkeypatch):
    root = __import__("pathlib").Path(__file__).parents[1]
    project_path = root / "Examples/EBS/multistage_fit/EBS_case_C_multistage.pyloco.json"
    historical = ProjectMetadata.load(project_path)
    sources = {
        "bpm_noise": root / "Examples/EBS/multistage_fit/data/measured_BPM_noise_loco_20260126_212023.h5",
        "dispersion": root / "Examples/EBS/multistage_fit/data/measured_dispersion_loco_20260126_221103.h5",
        "orm": root / "Examples/EBS/multistage_fit/data/ORM_for_pyLOCO_20260126_221131.h5",
    }
    window = MainWindow()
    for role, source in sources.items():
        window.measurement_role.setCurrentText(role)
        monkeypatch.setattr(
            "pyLOCO.gui.main_window.QFileDialog.getOpenFileName",
            lambda *args, _source=source, **kwargs: (str(_source), ""),
        )
        window.import_measurement()
    imported = {role: dataset.path for role, dataset in window.project.measurements.items()}
    assert window.project.modified is True
    monkeypatch.setattr(window, "_confirm_open_with_unsaved_changes", lambda: True)
    monkeypatch.setattr(window, "_choose_open_project_measurement_policy", lambda candidate: "keep")
    warnings = []
    monkeypatch.setattr("pyLOCO.gui.main_window.QMessageBox.warning", lambda *args: warnings.append(args[2]))
    window.open_project(project_path)
    assert warnings == []
    assert {role: dataset.path for role, dataset in window.project.measurements.items()} == imported
    assert [stage.name for stage in window.fit_recipe.stages] == [
        "Calibration", "BPM coupling", "Normal optics", "Skew and coupling"
    ]
    assert __import__("dataclasses").asdict(window.project.loco_config.machine_elements) == __import__("dataclasses").asdict(
        historical.loco_config.machine_elements
    )
    assert window.project.loco_config.bad_bpm_positions == [27, 231, 286]
    window.project.modified = False
    window.close()


def test_open_project_can_use_saved_measurements_or_cancel_without_changes(app, tmp_path, monkeypatch):
    historical, active = _case_c_project_with_copied_measurements(tmp_path)
    candidate_path = historical.save(tmp_path / "saved-measurements-project.pyloco.json")
    window = MainWindow(); window.project = active; window.project.modified = False; window._refresh_ui("handoff")
    active_signature = window._measurement_binding_signature(window.project)
    monkeypatch.setattr(window, "_choose_open_project_measurement_policy", lambda candidate: None)
    window.open_project(candidate_path)
    assert window.project.name == "new measurement handoff"
    assert window._measurement_binding_signature(window.project) == active_signature
    monkeypatch.setattr(window, "_choose_open_project_measurement_policy", lambda candidate: "saved")
    window.open_project(candidate_path)
    assert window.project.name == historical.name
    assert window._measurement_binding_signature(window.project) == window._measurement_binding_signature(historical)
    window.close()


def test_open_project_rejects_five_by_five_measurements_for_case_c_transactionally(app, tmp_path, monkeypatch):
    root = __import__("pathlib").Path(__file__).parents[1]
    historical = ProjectMetadata.load(root / "Examples/EBS/multistage_fit/EBS_case_C_multistage.pyloco.json")
    candidate_path = historical.save(tmp_path / "case-c-project.pyloco.json")
    manifest = root / "measurements/ebs-mdt-smoke-session.pyloco-session.json"
    if not manifest.is_file():
        pytest.skip("MDT smoke measurement session is not present in this development worktree")
    from pyLOCO.gui.suite import inspect_measurement_session
    handoff = inspect_measurement_session(manifest)
    active = ProjectMetadata(name="5H/5V active session")
    active.measurements = {
        role: ImportedDataset(role, str(source), source.suffix.lstrip("."), source.stat().st_size, dict(handoff.options[role]))
        for role, source in handoff.files.items()
    }
    active.measurement_session = dict(handoff.provenance)
    window = MainWindow(); window.project = active; window.project.modified = False; window._refresh_ui("handoff")
    before = window._measurement_binding_signature(window.project)
    failures = []
    monkeypatch.setattr(window, "_choose_open_project_measurement_policy", lambda candidate: "keep")
    monkeypatch.setattr(
        window, "_choose_open_for_editing",
        lambda candidate, errors: failures.extend(errors) or False,
    )
    window.open_project(candidate_path)
    assert window.project.name == "5H/5V active session"
    assert window._measurement_binding_signature(window.project) == before
    assert failures and any("expected (640, 64)" in error or "selects 32 horizontal correctors" in error
                            for error in failures)
    window.close()


def _smoke_measurement_project():
    root = __import__("pathlib").Path(__file__).parents[1]
    manifest = root / "measurements/ebs-mdt-smoke-session.pyloco-session.json"
    if not manifest.is_file():
        pytest.skip("MDT smoke measurement session is not present in this development worktree")
    from pyLOCO.gui.suite import inspect_measurement_session
    handoff = inspect_measurement_session(manifest)
    active = ProjectMetadata(name="5H/5V active session")
    active.measurements = {
        role: ImportedDataset(
            role, str(source), source.suffix.lstrip("."), source.stat().st_size,
            dict(handoff.options[role]),
        )
        for role, source in handoff.files.items()
    }
    active.measurement_session = dict(handoff.provenance)
    return active


def test_modern_ebs_session_metadata_is_retained_used_and_summarized(app):
    root = __import__("pathlib").Path(__file__).parents[1]
    manifest = root / "measurements/ebs-mdt-smoke-session.pyloco-session.json"
    if not manifest.is_file():
        pytest.skip("MDT smoke measurement session is not present in this development worktree")
    window = MainWindow()
    assert window.open_measurement_session(manifest) is True

    orm = window.project.measurements["orm"].options
    dispersion = window.project.measurements["dispersion"].options
    noise = window.project.measurements["bpm_noise"].options
    assert len(orm["bpm_names"]) == 320
    assert len(orm["horizontal_corrector_names"]) == 5
    assert len(orm["vertical_corrector_names"]) == 5
    assert orm["acquisition_metadata"]["response_matrix_unit"] == "m"
    assert orm["scaled"] is False
    assert np.allclose(orm["actual_kick_h_rad"], 1e-4)
    assert np.allclose(orm["actual_kick_v_rad"], 1e-4)
    assert dispersion["rf_step_hz"] == pytest.approx(-200.0)
    assert dispersion["acquisition_metadata"]["rf_difference_sign_convention"] == "negative_minus_positive"
    assert noise["datasets"] == {"horizontal": "Noise_BPMx", "vertical": "Noise_BPMy"}
    assert noise["acquisition_metadata"]["bpm_noise_unit"] == "m"
    assert window.project.measurement_session["session_id"] == "ebs-mdt-smoke-20261013"

    loaded = _load_measurements(
        {role: dataset.path for role, dataset in window.project.measurements.items()},
        {role: dataset.options for role, dataset in window.project.measurements.items()},
    )
    assert loaded["orm"].shape == (640, 10)
    assert loaded["eta_x"].shape == loaded["eta_y"].shape == (320,)
    assert loaded["noise_x"].shape == loaded["noise_y"].shape == (320,)
    assert window.project.loco_config.response_matrix.rfStep == pytest.approx(-200.0)
    assert window.project.loco_config.response_matrix.dkick_h == pytest.approx(1e-4)
    assert window.project.loco_config.response_matrix.dkick_v == pytest.approx(1e-4)
    summary = window.measurement_metadata_label.text()
    assert "Measurement metadata available" in summary
    assert "320 ordered BPM identities" in summary
    assert "5 H + 5 V correctors" in summary
    assert "signed RF step −200 Hz" in summary
    assert "Noise_BPMx / Noise_BPMy [m]" in summary
    window.project.modified = False
    window.close()


def test_open_incompatible_case_c_project_as_unsaved_editable_copy(app, tmp_path, monkeypatch):
    root = __import__("pathlib").Path(__file__).parents[1]
    historical_source = root / "Examples/EBS/multistage_fit/EBS_case_C_multistage.pyloco.json"
    historical = ProjectMetadata.load(historical_source)
    candidate_path = historical.save(tmp_path / "case-c-original.pyloco.json")
    original_bytes = candidate_path.read_bytes()
    active = _smoke_measurement_project()
    active_paths = {role: dataset.path for role, dataset in active.measurements.items()}
    active_session = dict(active.measurement_session)

    window = MainWindow(); window.project = active; window.project.modified = False; window._refresh_ui("handoff")
    monkeypatch.setattr(window, "_choose_open_project_measurement_policy", lambda candidate: "keep")
    monkeypatch.setattr(window, "_choose_open_for_editing", lambda candidate, errors: True)
    window.open_project(candidate_path)
    assert window._workspace.tabText(window._workspace.currentIndex()) == "Machine Components"
    assert "Current measurements preserved" in window.statusBar().currentMessage()

    assert {role: dataset.path for role, dataset in window.project.measurements.items()} == active_paths
    assert window.project.measurement_session == active_session
    assert window.project.path == ""
    assert window.project.base_directory == str(candidate_path.parent.resolve())
    assert window.project.modified is True
    assert candidate_path.read_bytes() == original_bytes
    assert [stage.name for stage in window.fit_recipe.stages] == [
        "Calibration", "BPM coupling", "Normal optics", "Skew and coupling"
    ]
    assert [stage.start_from for stage in window.fit_recipe.stages] == [
        "original_model", "previous_stage", "previous_stage", "previous_stage"
    ]
    assert window.project.completed_run.results_dir == ""
    assert window.project.fit_run_session == {}
    assert window.project.compatibility_validation["status"] == "incompatible"
    assert window.project.compatibility_validation["required"] is True
    assert window.run_loco_action.isEnabled() is False
    assert window.resume_fit_run_button.isEnabled() is False
    assert "fitting disabled" in window.compatibility_status_label.text().lower()

    saved_copy = tmp_path / "case-c-adapted.pyloco.json"
    monkeypatch.setattr(
        "pyLOCO.gui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(saved_copy), ""),
    )
    assert window.save_project_as() is True
    restored = ProjectMetadata.load(saved_copy)
    assert restored.compatibility_validation["status"] == "incompatible"
    assert {role: dataset.path for role, dataset in restored.measurements.items()} == active_paths
    assert candidate_path.read_bytes() == original_bytes
    window.project.modified = False
    window.close()


def test_editable_open_revalidates_after_compatible_machine_selection(app, tmp_path, monkeypatch):
    root = __import__("pathlib").Path(__file__).parents[1]
    historical = ProjectMetadata.load(root / "Examples/EBS/multistage_fit/EBS_case_C_multistage.pyloco.json")
    candidate_path = historical.save(tmp_path / "case-c-project.pyloco.json")
    active = _smoke_measurement_project()
    window = MainWindow(); window.project = active; window.project.modified = False; window._refresh_ui("handoff")
    monkeypatch.setattr(window, "_choose_open_project_measurement_policy", lambda candidate: "keep")
    monkeypatch.setattr(window, "_choose_open_for_editing", lambda candidate, errors: True)
    window.open_project(candidate_path)

    elements = window.project.loco_config.machine_elements
    elements.horizontal_corrector_ords = elements.horizontal_corrector_ords[:5]
    elements.vertical_corrector_ords = elements.vertical_corrector_ords[:5]
    # Empty selections are valid for parameter families that are not fitted.
    elements.quadrupole_tilt_ords = []
    for stage in window.fit_recipe.stages:
        mapping = stage.configuration.get("backend_mapping", stage.configuration)
        fit_list = mapping.get("FitInitConfig", {}).get("fit_list") or mapping.get("LOCOOptions", {}).get("fit_list") or []
        assert "quads_tilt" not in fit_list
    window.project.machine_configuration_override = window._machine_override_from_config(window.project.loco_config)
    window._apply_machine_override_to_recipe(window.fit_recipe, window.project.machine_configuration_override)
    window._mark_compatibility_stale("Machine selections changed; validation is required.")
    window._load_config_to_widgets()
    window._refresh_ui("edited")
    assert window.run_loco_action.isEnabled() is False

    messages = []
    monkeypatch.setattr(
        "pyLOCO.gui.main_window.QMessageBox.information",
        lambda *args: messages.append((args[1], args[2])),
    )
    monkeypatch.setattr(
        window, "_show_project_open_validation_error",
        lambda candidate, errors: pytest.fail("configuration should revalidate: " + " | ".join(errors)),
    )
    window.validate_project()
    assert window.project.compatibility_validation["status"] == "validated"
    assert window.run_loco_action.isEnabled() is True
    assert window.resume_fit_run_button.isEnabled() is True
    assert messages and "passed" in messages[-1][1].lower()
    window.project.modified = False
    window.close()


def test_editable_open_preserves_distinct_stage_machine_selections(app, tmp_path, monkeypatch):
    from copy import deepcopy
    root = __import__("pathlib").Path(__file__).parents[1]
    candidate = ProjectMetadata.load(root / "Examples/EBS/multistage_fit/EBS_case_C_multistage.pyloco.json")
    probe = MainWindow()
    recipe = probe._recipe_from_project(candidate)
    probe.close()
    assert recipe is not None
    mapping = recipe.stages[1].configuration.get("backend_mapping", recipe.stages[1].configuration)
    mapping["MachineElements"]["horizontal_corrector_ords"] = mapping["MachineElements"]["horizontal_corrector_ords"][:-1]
    candidate.fit_recipe = recipe.to_dict()
    candidate_path = candidate.save(tmp_path / "different-stage-selections.pyloco.json")
    active = _smoke_measurement_project()
    candidate.measurements = deepcopy(active.measurements)
    candidate.measurement_session = deepcopy(active.measurement_session)
    window = MainWindow()
    before = [deepcopy(stage.configuration) for stage in window._recipe_from_project(candidate).stages]
    monkeypatch.setattr(window, "_choose_adapted_machine_policy", lambda loaded: "recipe")
    assert window._prepare_incompatible_working_copy(candidate, ["ORM dimensions differ."], str(candidate_path))
    after = [stage.configuration for stage in window._recipe_from_project(candidate).stages]
    assert after == before
    assert candidate.recipe_machine_policy == "recipe"
    assert candidate.machine_configuration_override == {}
    window.close()


def test_incomplete_recipe_identity_reports_configuration_problem_not_zero_dimensions(app, tmp_path, monkeypatch):
    from copy import deepcopy
    root = __import__("pathlib").Path(__file__).parents[1]
    historical, active = _case_c_project_with_copied_measurements(tmp_path)
    incomplete = deepcopy(historical)
    incomplete.name = "Incomplete stage-only project"
    incomplete.fit_recipe["data_requirements"] = {}
    incomplete_path = incomplete.save(tmp_path / "incomplete-stage-only.pyloco.json")
    window = MainWindow(); window.project = active; window.project.modified = False; window._refresh_ui("handoff")
    before = window.project.to_dict()
    captured = []
    monkeypatch.setattr(window, "_choose_open_project_measurement_policy", lambda candidate: "keep")
    monkeypatch.setattr(window, "_choose_open_for_editing", lambda candidate, errors: captured.extend(errors) or False)
    window.open_project(incomplete_path)
    assert window.project.to_dict() == before
    assert len(captured) == 1
    assert captured[0].startswith("MISSING_MEASUREMENT_PROVENANCE:")
    assert "stage 1 (Calibration): 320 BPMs, 32 H and 32 V correctors" in captured[0]
    assert "expected (0, 0)" not in captured[0]
    window.close()


def test_project_open_dialogs_are_short_with_expandable_diagnostics(app, tmp_path, monkeypatch):
    historical, active = _case_c_project_with_copied_measurements(tmp_path)
    window = MainWindow(); window.project = active; window.project.modified = False; window._refresh_ui("handoff")
    errors = [
        "MISSING_MEASUREMENT_PROVENANCE: Missing BPM/corrector identities and pinned checksums."
    ]
    content = window._project_open_error_content(historical, errors)
    assert content["title"] == "Measurements cannot be verified"
    assert len(content["summary"]) < 180
    assert "remain unchanged" in content["information"]
    assert "Selected project:" in content["details"]
    assert "Validation failures:" in content["details"]
    content = window._project_load_error_content(
        str(tmp_path / "broken.pyloco.json"), ValueError("invalid JSON")
    )
    assert content["title"] == "Unable to open FIT project"
    assert "remain unchanged" in content["information"]
    assert "ValueError" in content["details"]
    window.close()


def test_exact_ebs_my_test_project_opens_for_editing_with_case_c_measurements(app, tmp_path, monkeypatch):
    exact_project = __import__("pathlib").Path(
        os.environ.get("PYLOCO_EBS_MY_TEST_PROJECT", "/external/ebs_my_test.pyloco.json")
    )
    if not exact_project.is_file():
        pytest.skip("User's exact ebs_my_test.pyloco.json is not available on this host")
    original_bytes = exact_project.read_bytes()
    _historical, active = _case_c_project_with_copied_measurements(tmp_path)
    active_paths = {role: dataset.path for role, dataset in active.measurements.items()}
    window = MainWindow(); window.project = active; window.project.modified = False; window._refresh_ui("handoff")
    reached = []
    monkeypatch.setattr(window, "_choose_open_project_measurement_policy", lambda candidate: "keep")
    monkeypatch.setattr(
        window, "_choose_open_for_editing",
        lambda candidate, errors: reached.extend(errors) or True,
    )
    window.open_project(exact_project)
    assert reached and reached[0].startswith("MISSING_MEASUREMENT_PROVENANCE:")
    assert {role: dataset.path for role, dataset in window.project.measurements.items()} == active_paths
    assert [stage.name for stage in window.fit_recipe.stages] == ["Stage 2"]
    elements = window.project.loco_config.machine_elements
    assert (len(elements.bpm_ords), len(elements.horizontal_corrector_ords), len(elements.vertical_corrector_ords)) == (320, 32, 32)
    assert window.project.path == ""
    assert window.project.compatibility_validation["status"] == "incompatible"
    assert window.run_loco_action.isEnabled() is False
    assert window.resume_fit_run_button.isEnabled() is False
    assert exact_project.read_bytes() == original_bytes
    window.project.modified = False
    window.close()


def test_malformed_stage_configuration_is_not_offered_as_editable_mismatch(app, tmp_path, monkeypatch):
    from copy import deepcopy
    historical, active = _case_c_project_with_copied_measurements(tmp_path)
    malformed = deepcopy(historical)
    malformed.fit_recipe["data_requirements"] = {}
    mapping = malformed.fit_recipe["stages"][0]["configuration"]["backend_mapping"]
    mapping.pop("RMConfig")
    malformed_path = malformed.save(tmp_path / "malformed-stage.pyloco.json")
    window = MainWindow(); window.project = active; window.project.modified = False; window._refresh_ui("handoff")
    captured = []
    monkeypatch.setattr(window, "_choose_open_project_measurement_policy", lambda candidate: "keep")
    monkeypatch.setattr(window, "_show_project_open_validation_error", lambda candidate, errors: captured.extend(errors))
    monkeypatch.setattr(
        window, "_choose_open_for_editing",
        lambda *args: (_ for _ in ()).throw(AssertionError("malformed projects must not enter editing mode")),
    )
    window.open_project(malformed_path)
    assert captured and captured[0].startswith("MALFORMED_PROJECT_CONFIGURATION:")
    assert window.project.name == active.name
    window.close()


def test_saved_fit_run_identity_mismatch_is_detected(app, tmp_path):
    from pyLOCO.gui.fit_workflow import FitRunSession
    measurement = tmp_path / "orm.h5"; measurement.write_bytes(b"placeholder")
    run_session = FitRunSession(
        recipe={}, lattice_path="lattice.mat", lattice_checksum="checksum",
        measurements={"orm": str(measurement)},
        measurement_identity={"session_id": "old-session"},
    )
    session_path = run_session.save(tmp_path / "fit-run-session.json")
    project = ProjectMetadata(name="identity mismatch")
    project.measurements["orm"] = ImportedDataset("orm", str(measurement), "h5")
    project.measurement_session = {"session_id": "new-session"}
    project.fit_run_session = {"path": str(session_path), "completed_stages": 0}
    window = MainWindow()
    assert window._fit_run_measurements_are_stale(project) is True
    window.close()


def test_open_project_without_active_measurements_does_not_prompt(app, tmp_path, monkeypatch):
    target = ProjectMetadata(name="ordinary open").save(tmp_path / "ordinary.pyloco.json")
    window = MainWindow()
    monkeypatch.setattr(
        window, "_choose_open_project_measurement_policy",
        lambda candidate: (_ for _ in ()).throw(AssertionError("dialog should not open")),
    )
    window.open_project(target)
    assert window.project.name == "ordinary open"
    window.close()


def test_plot_parameter_and_summary_exports(app, tmp_path, monkeypatch):
    result_dir = tmp_path / "results"; _synthetic_results(result_dir); loader = ResultsLoader(result_dir)
    plot = PlotCanvas(); plot.figure.add_subplot(111).plot([0, 1], [1, 0])
    png = tmp_path / "plot.png"
    monkeypatch.setattr("pyLOCO.gui.results.plot_canvas.QFileDialog.getSaveFileName", lambda *a, **k: (str(png), ""))
    assert plot.save_plot() == str(png) and png.exists()
    parameters = ParametersView(); parameters.set_loader(loader); exported = tmp_path / "parameters.json"
    monkeypatch.setattr("pyLOCO.gui.results.parameters_view.QFileDialog.getSaveFileName", lambda *a, **k: (str(exported), ""))
    parameters._export("json"); assert json.loads(exported.read_text())[0]["block"] == "hbpm_gain"
    summary = RunSummaryView(); summary.set_loader(loader); summary_path = tmp_path / "summary.json"
    rendered=summary.text.toPlainText()
    assert "Run overview" in rendered and "Measurement inputs" in rendered and "Fit results" in rendered
    assert not rendered.lstrip().startswith("{")
    monkeypatch.setattr("pyLOCO.gui.results.run_summary_view.QFileDialog.getSaveFileName", lambda *a, **k: (str(summary_path), ""))
    summary._export("json"); assert json.loads(summary_path.read_text())["final_chi2"] == 2.0


def test_shared_plot_export_can_omit_title_without_changing_the_screen_plot(app,tmp_path,monkeypatch):
    plot=PlotCanvas(); axis=plot.figure.add_subplot(111); axis.set_title("Scientific plot title")
    destination=tmp_path/"without-title.png"; observed=[]
    monkeypatch.setattr("pyLOCO.gui.results.plot_canvas.QFileDialog.getSaveFileName",lambda *a,**k:(str(destination),""))
    monkeypatch.setattr(plot.figure,"savefig",lambda *a,**k:observed.append(axis.get_title()))
    assert plot.include_title.isChecked()
    plot.include_title.setChecked(False); assert plot.save_plot()==str(destination)
    assert observed==[""] and axis.get_title()=="Scientific plot title"
    plot.include_title.setChecked(True); plot.save_plot()
    assert observed[-1]=="Scientific plot title"


def test_restored_run_exports_exact_fitted_data_and_jacobian(app, tmp_path, monkeypatch):
    result_dir = tmp_path / "results"; _synthetic_results(result_dir)
    matrix = np.arange(30.0).reshape(10, 3)
    np.savez_compressed(result_dir / "jacobian.npz", matrix=matrix)
    (result_dir / "jacobian_metadata.json").write_text(json.dumps({"shape": [10, 3], "parameter_blocks": {"hbpm_gain": {"start": 0, "stop": 2}}}))
    loader = ResultsLoader(result_dir)
    parameters = ParametersView(); parameters.set_loader(loader)
    exported = tmp_path / "fitted.json"
    monkeypatch.setattr("pyLOCO.gui.results.parameters_view.QFileDialog.getSaveFileName", lambda *a, **k: (str(exported), ""))
    parameters._export("json")
    rows = json.loads(exported.read_text())
    assert len([row for row in rows if row["block"] == "hbpm_gain"]) == 2
    from pyLOCO.gui.results.svd_view import SvdView
    view = SvdView(); view.set_loader(loader); assert view.save_jacobian.isEnabled()
    jac_export = tmp_path / "exported_jacobian.npz"
    monkeypatch.setattr("pyLOCO.gui.results.svd_view.QFileDialog.getSaveFileName", lambda *a, **k: (str(jac_export), ""))
    view._save_jacobian()
    with np.load(jac_export, allow_pickle=False) as archive:
        assert np.array_equal(archive["matrix"], matrix)  # legacy dense artifact remains exportable
    assert json.loads((tmp_path / "exported_jacobian_metadata.json").read_text())["shape"] == [10, 3]
