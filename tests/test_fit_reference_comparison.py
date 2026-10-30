from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pyLOCO.gui.backend import LocoRunRequest
from pyLOCO.gui.reference_comparison import calculate_reference_comparison
from pyLOCO.gui.reference_comparison import ReferenceComparisonResult


ROOT = Path(__file__).parents[1]
CASE_C_RUN = (
    ROOT / "Examples/EBS/multistage_fit/results/"
    "ESRF-EBS_historical_Case_C___four-stage_fit-20261016-105359"
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _case_c_request() -> LocoRunRequest:
    path = CASE_C_RUN / "run_request.json"
    if not path.is_file():
        pytest.skip("The validated Case C run artifact is unavailable")
    return LocoRunRequest(**json.loads(path.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def case_c_result():
    pytest.importorskip("at")
    return calculate_reference_comparison(_case_c_request())


def test_case_c_reference_orm_matches_equivalent_saved_initial_model(case_c_result):
    artifact = CASE_C_RUN / "model_orm_initial.h5"
    with h5py.File(artifact, "r") as stream:
        saved = np.asarray(stream["response_matrix"])
    # The saved file includes RF response as its final column.  ORM agreement
    # is checked independently so RF-orbit and physical-dispersion quantities
    # can never be conflated by this assertion.
    np.testing.assert_allclose(case_c_result.model_orm, saved[:, :64], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        case_c_result.measured_orm - case_c_result.model_orm,
        case_c_result.measured_orm - saved[:, :64], rtol=0.0, atol=0.0,
    )


def test_case_c_rf_response_and_physical_dispersion_are_distinct_signed_quantities(case_c_result):
    assert case_c_result.measured_rf is not None
    assert case_c_result.physical_measured is not None
    coefficient = case_c_result.provenance["physical_dispersion_coefficient"]
    assert coefficient == pytest.approx(-151.92331126998323)
    np.testing.assert_allclose(
        case_c_result.physical_measured,
        case_c_result.measured_rf * coefficient,
    )
    assert case_c_result.provenance["rf_quantity"] == "signed RF-induced orbit difference [m]"
    assert case_c_result.provenance["residual_definition"] == "Measured − reference model"
    assert set(case_c_result.rf_statistics) == {"Horizontal", "Vertical"}
    assert case_c_result.rf_statistics["Horizontal"]["measured_rms"] == pytest.approx(0.00033532549476537947)
    assert case_c_result.rf_statistics["Horizontal"]["model_rms"] == pytest.approx(0.0006250957672231784)
    assert case_c_result.rf_statistics["Horizontal"]["residual_rms"] == pytest.approx(0.0002899829492330071)


def test_reference_comparison_reports_cross_plane_objective_mask(case_c_result):
    request = _case_c_request()
    request.backend_mapping["LOCOOptions"]["remove_coupling_"] = True
    masked = calculate_reference_comparison(request)
    assert masked.cross_plane_excluded is True
    assert set(masked.statistics) == {"Global", "H→X", "V→X", "H→Y", "V→Y"}
    # Displayed statistics remain available even when cross-plane blocks are
    # excluded from the objective.
    assert np.isfinite(masked.statistics["H→Y"]["residual_rms"])
    assert np.isfinite(masked.statistics["V→X"]["residual_rms"])


def test_reference_comparison_never_calls_optimizer(monkeypatch):
    pytest.importorskip("at")
    import pyLOCO.pyloco as pyloco_module

    monkeypatch.setattr(
        pyloco_module, "pyloco",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("optimizer called")),
    )
    result = calculate_reference_comparison(_case_c_request())
    assert result.model_orm.shape == result.measured_orm.shape


def test_reference_comparison_does_not_require_fitted_quadrupole_selection():
    pytest.importorskip("at")
    request=_case_c_request()
    request.backend_mapping["FitInitConfig"]["fit_list"]=["quads"]
    request.backend_mapping["MachineElements"]["normal_quadrupole_ords"]=[]
    result=calculate_reference_comparison(request)
    assert result.model_orm.shape==result.measured_orm.shape


def test_gui_exposes_distinct_pre_and_post_fit_comparison_actions(app):
    from pyLOCO.gui.main_window import MainWindow

    window = MainWindow()
    assert window.compare_reference_action.text() == "Compare imported measurements with initial model…"
    assert window.compare_orms_action.text() == "Compare measurements with fitted model…"
    assert "optimizer" in window.compare_reference_action.toolTip().lower()
    assert window.compare_reference_button.text() == "Compare imported measurements with initial model…"
    assert window.pre_fit_checks_group.title() == "Pre-FIT measurement checks"
    assert "selecting a stage edits its configuration" in window.fit_workflow_hint.text().lower()
    assert "entire validated recipe" in window.fit_workflow_hint.text().lower()
    toolbar = window.findChild(__import__("PySide6.QtWidgets", fromlist=["QToolBar"]).QToolBar, "mainToolbar")
    assert window.compare_reference_action not in toolbar.actions()
    assert window.compare_orms_action not in toolbar.actions()
    assert window.results_workspace.orm.side_by_side_button.text() == "Compare measured and fitted model…"
    window.project.modified = False
    window.close()


def test_reference_window_displays_physical_dispersion_in_millimetres(app):
    from pyLOCO.gui.widgets.reference_comparison import ReferenceComparisonWindow

    result = ReferenceComparisonResult(
        measured_orm=np.zeros((4, 2)), model_orm=np.zeros((4, 2)),
        measured_rf=np.zeros(4), model_rf=np.zeros(4),
        physical_measured=np.array([.001, .002, .003, .004]),
        physical_model=np.zeros(4), n_bpms=2, n_hcors=1, n_vcors=1,
        cross_plane_excluded=False, statistics={}, rf_statistics={}, provenance={},
    )
    window = ReferenceComparisonWindow(result)
    window.rf_quantity_selector.setCurrentIndex(1)

    assert window.rf_quantity_selector.currentText() == "Physical dispersion D [mm]"
    measured_rms, unit = window.rf_statistics_table.item(0, 0).text().split()
    assert unit == "mm"
    assert float(measured_rms) == pytest.approx(np.sqrt((1.0 ** 2 + 2.0 ** 2) / 2.0))
    assert any(
        "converted from metres to millimetres" in label.text()
        for label in window.findChildren(__import__("PySide6.QtWidgets", fromlist=["QLabel"]).QLabel)
    )
    window.close()


def test_case_c_post_fit_comparison_separates_rf_and_replays_exclusions():
    from pyLOCO.gui.results.results_loader import ResultsLoader

    if not (CASE_C_RUN / "run_request.json").is_file():
        pytest.skip("The validated Case C run artifact is unavailable")
    loader = ResultsLoader(CASE_C_RUN).for_iteration(2)
    measured, model, context = loader.orm_comparison_inputs()
    assert measured.shape == model.shape == (634, 64)
    assert context["bpm_count"] == 317
    assert context["horizontal_correctors"] == 32
    assert context["vertical_correctors"] == 32
    assert context["rf_column_separated"] is True
    assert context["iteration"] == 2


def test_post_fit_comparison_fails_closed_without_recorded_device_inventory(tmp_path):
    from pyLOCO.gui.results.results_loader import ResultsLoader

    (tmp_path / "run_request.json").write_text(json.dumps({
        "project_name": "unbound", "measurements": {}, "backend_mapping": {}
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="ordered BPM"):
        ResultsLoader(tmp_path).orm_comparison_inputs()


def test_reference_window_separates_rf_planes_and_collapses_technical_details(app, case_c_result):
    from pyLOCO.gui.widgets.reference_comparison import ReferenceComparisonWindow

    window = ReferenceComparisonWindow(case_c_result)
    assert window.windowTitle() == "Compare imported measurements with initial model"
    assert window.rf_plane_selector.count() == 2
    assert window.rf_plane_selector.itemText(0) == "Horizontal"
    assert window.rf_plane_selector.itemText(1) == "Vertical"
    assert window.rf_quantity_selector.itemText(0) == "RF orbit response [m]"
    assert window.rf_quantity_selector.itemText(1) == "Physical dispersion D [mm]"
    assert window.rf_statistics_table.columnCount() == 3
    assert window.rf_statistics_table.rowCount() == 2
    assert window.rf_statistics_table.verticalHeaderItem(0).text() == "Horizontal (X)"
    assert window.rf_statistics_table.verticalHeaderItem(1).text() == "Vertical (Y)"
    assert window.rf_statistics_table.verticalScrollBarPolicy() == __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.ScrollBarAlwaysOff
    assert window.technical_details_group.isChecked() is False
    rf_horizontal = window.rf_statistics_table.item(0, 2).text()
    rf_vertical = window.rf_statistics_table.item(1, 2).text()
    window.rf_quantity_selector.setCurrentIndex(1)
    assert window.rf_statistics_table.item(0, 2).text() != rf_horizontal
    assert window.rf_statistics_table.item(1, 2).text() != rf_vertical
    expected_horizontal_mm = 1000.0 * np.sqrt(np.mean(
        (case_c_result.physical_measured[:case_c_result.n_bpms]
         - case_c_result.physical_model[:case_c_result.n_bpms]) ** 2
    ))
    displayed_value, displayed_unit = window.rf_statistics_table.item(0, 2).text().split()
    assert displayed_unit == "mm"
    assert float(displayed_value) == pytest.approx(expected_horizontal_mm)
    window.rf_plane_selector.setCurrentIndex(1)
    assert window.rf_statistics_table.item(0, 2).text().endswith(" mm")
    window.close()
