from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QAbstractButton, QLabel, QScrollArea
from PySide6.QtCore import Qt

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
from pyLOCO.gui.main_window import ElementSelectionDialog, ExclusionSelectionDialog, MainWindow, ScientificDoubleSpinBox
from pyLOCO.gui.results.parameters_view import ParametersView
from pyLOCO.gui.results.plot_canvas import PlotCanvas
from pyLOCO.gui.results.results_loader import ResultsLoader
from pyLOCO.gui.results.results_workspace import ResultsWorkspace
from pyLOCO.gui.results.run_summary_view import RunSummaryView


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


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
    dialog.auto_radio.setChecked(True)
    assert not dialog.type_edit.isEnabled() and not dialog.manual_edit.isEnabled()
    dialog.name_file_radio.setChecked(True)
    assert dialog.name_file_edit.isEnabled() and dialog.name_attribute.isEnabled()
    assert not dialog.file_edit.isEnabled() and not dialog.pattern_edit.isEnabled()
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
    assert len(rows) == len(buttons) == 6
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


def test_spinbox_wheel_requires_focus(app):
    class Event:
        ignored = False
        def ignore(self): self.ignored = True
    spin = ScientificDoubleSpinBox(); event = Event(); spin.clearFocus(); spin.wheelEvent(event)
    assert event.ignored


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
    monkeypatch.setattr("pyLOCO.gui.results.run_summary_view.QFileDialog.getSaveFileName", lambda *a, **k: (str(summary_path), ""))
    summary._export("json"); assert json.loads(summary_path.read_text())["final_chi2"] == 2.0


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
