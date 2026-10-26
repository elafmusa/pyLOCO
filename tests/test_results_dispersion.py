import json

import h5py
import numpy as np
import pytest
from types import SimpleNamespace
from PySide6.QtWidgets import QApplication

from pyLOCO.gui.backend import _save_optics_results
from pyLOCO.gui.results.optics_view import OpticsView
from pyLOCO.gui.results.orm_view import OrmView
from pyLOCO.gui.results.results_loader import ResultsLoader, notebook_physical_dispersion
from pyLOCO.gui.results.results_workspace import ResultsWorkspace


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _request(path, *, included=False, measurements=None, lattice_path=""):
    path.write_text(json.dumps({
        "lattice_path": lattice_path,
        "measurements": measurements or {},
        "backend_mapping": {"LOCOOptions": {"includeDispersion": included}},
    }))


def _persisted(path, *, included):
    measured_x = np.array([1.0, 2.0, 4.0]) * 1e-3
    measured_y = np.array([-2.0, 0.0, 3.0]) * 1e-3
    initial_x = np.array([0.0, 2.5, 3.0]) * 1e-3
    initial_y = np.array([-1.0, 1.0, 2.0]) * 1e-3
    fitted_x = np.array([0.5, 2.0, 3.5]) * 1e-3
    fitted_y = np.array([-1.5, 0.5, 2.5]) * 1e-3
    np.savez_compressed(
        path / "optics_results.npz",
        dispersion_in_fit=included,
        dispersion_diagnostic_available=True,
        dispersion_s=np.array([10.0, 20.0, 30.0]),
        dispersion_x_measured=measured_x, dispersion_y_measured=measured_y,
        dispersion_x_initial=initial_x, dispersion_y_initial=initial_y,
        dispersion_x_fitted=fitted_x, dispersion_y_fitted=fitted_y,
        s=np.array([0.0, 10.0, 20.0]),
        beta_x_reference=np.array([10.0, 12.0, 14.0]),
        beta_y_reference=np.array([8.0, 9.0, 10.0]),
        beta_x_initial=np.array([11.0, 11.4, 14.7]),
        beta_y_initial=np.array([7.6, 9.9, 10.2]),
        beta_x_fitted=np.array([10.2, 12.24, 13.86]),
        beta_y_fitted=np.array([8.08, 8.82, 10.3]),
        beta_beating_x_initial=np.array([.1, -.05, .05]),
        beta_beating_y_initial=np.array([-.05, .1, .02]),
        beta_beating_x=np.array([.02, .02, -.01]),
        beta_beating_y=np.array([.01, -.02, .03]),
    )
    return measured_x, measured_y, initial_x, initial_y, fitted_x, fitted_y


@pytest.mark.parametrize("included", [True, False])
def test_dispersion_diagnostic_available_independently_of_fit_objective(tmp_path, included):
    _request(tmp_path / "run_request.json", included=included)
    expected = _persisted(tmp_path, included=included)
    loader = ResultsLoader(tmp_path)
    assert loader.dispersion_included is included
    assert loader.dispersion_data is not None
    np.testing.assert_array_equal(loader.dispersion_data["x"]["measured"], expected[0])


def test_dispersion_excluded_and_measurement_unavailable(tmp_path):
    _request(tmp_path / "run_request.json", included=False)
    loader = ResultsLoader(tmp_path)
    assert loader.dispersion_data is None
    assert loader.dispersion_unavailable_reason == "Measured dispersion is not available for this run."


def test_dispersion_reference_or_fitted_lattice_unavailable(tmp_path):
    measurement = tmp_path / "dispersion.h5"
    with h5py.File(measurement, "w") as handle:
        handle["measured_eta_x"] = [0.0]
        handle["measured_eta_y"] = [0.0]
    _request(tmp_path / "run_request.json", included=False, measurements={"dispersion": str(measurement)}, lattice_path="missing.mat")
    loader = ResultsLoader(tmp_path)
    assert loader.dispersion_data is None
    assert loader.dispersion_unavailable_reason == "Initial/reference lattice is unavailable."


def test_dispersion_statistics_match_direct_numpy(tmp_path):
    _request(tmp_path / "run_request.json", included=False)
    measured_x, _, initial_x, _, fitted_x, _ = _persisted(tmp_path, included=False)
    stats = ResultsLoader(tmp_path).dispersion_statistics["x"]
    before, after = measured_x - initial_x, measured_x - fitted_x
    assert stats["rms_before"] == pytest.approx(np.sqrt(np.mean(before**2)))
    assert stats["rms_after"] == pytest.approx(np.sqrt(np.mean(after**2)))
    assert stats["mean_before"] == pytest.approx(np.mean(before))
    assert stats["mean_after"] == pytest.approx(np.mean(after))
    assert stats["min_before"] == pytest.approx(np.min(before))
    assert stats["max_before"] == pytest.approx(np.max(before))
    assert stats["max_abs_after"] == pytest.approx(np.max(np.abs(after)))
    assert stats["improvement"] == pytest.approx(100 * (1 - np.sqrt(np.mean(after**2)) / np.sqrt(np.mean(before**2))))


def test_optics_results_are_plane_separated_and_use_saved_arrays(app, tmp_path):
    _request(tmp_path / "run_request.json", included=True)
    request=json.loads((tmp_path/"run_request.json").read_text())
    request["backend_mapping"]["MachineElements"]={"bpm_ords":[0,2]}
    (tmp_path/"run_request.json").write_text(json.dumps(request))
    measured_x, measured_y, initial_x, initial_y, fitted_x, fitted_y = _persisted(tmp_path, included=True)
    loader = ResultsLoader(tmp_path)
    view = OpticsView(); view.resize(1200, 800); view.show(); view.set_loader(loader); app.processEvents()

    assert [view.tabs.tabText(i) for i in range(view.tabs.count())] == ["βx", "βy", "Dispersion X", "Dispersion Y"]
    assert view.beta_pages["x"] is not view.beta_pages["y"]
    assert view.dispersion_pages["x"] is not view.dispersion_pages["y"]
    view.resize(1200, 520); app.processEvents()
    assert view.beta_pages["x"].verticalScrollBar().maximum() > 0
    assert view.dispersion_pages["x"].verticalScrollBar().maximum() > 0
    np.testing.assert_allclose(
        view.dispersion_plots["x"]["residuals"].figure.axes[0].lines[0].get_ydata(),
        1000.0 * (measured_x - initial_x),
    )
    np.testing.assert_allclose(
        view.dispersion_plots["x"]["initial_residual"].figure.axes[0].lines[0].get_ydata(),
        1000.0 * (measured_x - initial_x),
    )
    np.testing.assert_allclose(
        view.dispersion_plots["y"]["fitted_residual"].figure.axes[0].lines[0].get_ydata(),
        1000.0 * (measured_y - fitted_y),
    )
    assert view.dispersion_tables["x"].rowCount() == 2
    assert view.beta_tables["x"].rowCount() == 2
    assert view.beta_tables["x"].horizontalHeaderItem(4).text()=="RMS beta beating [%]"
    assert "RMS beta beating" in view.beta_messages["x"].text()
    np.testing.assert_allclose(view.beta_plots["x"]["curves_bpms"].figure.axes[0].lines[0].get_xdata(),[0,1])
    assert view.beta_plots["x"]["curves_bpms"].figure.axes[0].get_xlabel()=="Selected BPM index (zero-based)"
    assert view.beta_plots["x"]["curves_bpms"].figure.axes[0].get_title()=="βx at 2 mapped BPMs"
    bpm_beating_lines=[line for line in view.beta_plots["x"]["beating_bpms"].figure.axes[0].lines if "β beating" in line.get_label()]
    np.testing.assert_allclose(bpm_beating_lines[-1].get_xdata(),[0,1])
    assert loader.beta_beating_statistics["x"]["initial"]["rms"] == pytest.approx(
        100.0 * np.sqrt(np.mean(np.array([.1, -.05, .05]) ** 2))
    )
    assert loader.beta_beating_statistics["y"]["fitted"]["max_abs"] == pytest.approx(3.0)
    for collection in (view.beta_plots, view.dispersion_plots):
        for plots in collection.values():
            for plot in plots.values():
                assert plot.save_button.isEnabled()
                # Optics plots remain visible in a laptop-height Results
                # viewport and expand naturally when more space is present.
                assert plot.minimumHeight() >= 120
                assert plot.canvas.height() > 0
    view.close()


def test_notebook_physical_dispersion_preserves_signed_pyat_convention():
    response = np.array([1.0e-4, -2.0e-4])
    at_eta = -8.0e-5
    positive_step = notebook_physical_dispersion(
        response, at_slip_factor=at_eta, rf_frequency_hz=500.0e6,
        signed_rf_step_hz=200.0,
    )
    negative_step = notebook_physical_dispersion(
        -response, at_slip_factor=at_eta, rf_frequency_hz=500.0e6,
        signed_rf_step_hz=-200.0,
    )
    np.testing.assert_allclose(positive_step, response * at_eta * 500.0e6 / 200.0)
    np.testing.assert_allclose(negative_step, positive_step)


def test_physical_dispersion_view_is_explicit_and_residual_is_measured_minus_model(app, tmp_path):
    _request(tmp_path / "run_request.json", included=True)
    _persisted(tmp_path, included=True)
    loader = ResultsLoader(tmp_path)
    data = loader.dispersion_data
    physical = {
        plane: {key: np.asarray(values[key]) * 10.0 for key in ("measured", "initial", "fitted")}
        for plane, values in (("x", data["x"]), ("y", data["y"]))
    }
    physical.update({"at_slip_factor":-1e-4,"rf_momentum_coefficient":1e-4,
                     "rf_frequency_hz":500e6,"signed_rf_step_hz":-200.0,
                     "definition":"test","residual_definition":"measured minus model"})
    data["physical"] = physical
    view = OpticsView(); view.resize(1200,800); view.show(); view.set_loader(loader); app.processEvents()
    assert view.dispersion_quantity["x"].currentData() == "physical"
    assert "Physical dispersion is a derived diagnostic" in view.dispersion_messages["x"].text()
    np.testing.assert_allclose(
        view.dispersion_plots["x"]["measured"].figure.axes[0].lines[0].get_ydata(),
        1000.0 * physical["x"]["measured"],
    )
    np.testing.assert_allclose(
        view.dispersion_plots["x"]["fitted_residual"].figure.axes[0].lines[0].get_ydata(),
        1000.0 * (physical["x"]["measured"] - physical["x"]["fitted"]),
    )
    assert view.dispersion_plots["x"]["fitted_residual"].figure.axes[0].get_legend().texts[0].get_text() == "Measured − fitted model"

    view.dispersion_quantity["x"].setCurrentIndex(
        view.dispersion_quantity["x"].findData("rf_orbit_response")
    )
    app.processEvents()
    np.testing.assert_allclose(
        view.dispersion_plots["x"]["fitted_residual"].figure.axes[0].lines[0].get_ydata(),
        1000.0 * (data["x"]["measured"] - data["x"]["fitted"]),
    )
    assert "measured RF orbit difference − model RF orbit difference" in view.dispersion_messages["x"].text()
    view.close()


def test_orm_results_residual_arrays_and_plot_are_measured_minus_model(app, tmp_path):
    measured = np.array([[4.0, -2.0], [1.0, 8.0]]) * 1e-6
    initial = np.array([[1.0, -5.0], [2.0, 3.0]]) * 1e-6
    fitted = np.array([[3.5, -1.5], [1.25, 7.0]]) * 1e-6
    loader = ResultsLoader(tmp_path)
    loader._cache.update(
        measured_orm=measured,
        initial_orm=initial,
        fitted_orm=fitted,
    )

    np.testing.assert_allclose(loader.residual_before, measured - initial)
    np.testing.assert_allclose(loader.residual_after, measured - fitted)

    view = OrmView()
    view.resize(1200, 800)
    view.show()
    view.set_loader(loader)
    view.selector.setCurrentIndex(view.selector.findData("residual_after"))
    app.processEvents()
    plotted = np.asarray(view.plot.figure.axes[0].images[0].get_array())
    np.testing.assert_allclose(plotted, measured - fitted)
    assert "measured − fitted model" in view.metrics.toolTip()
    view.close()


def test_results_separates_metadata_verified_rf_column_for_initial_and_fitted_orm(tmp_path):
    measured = np.arange(24, dtype=float).reshape(6, 4) * 1e-6
    initial = measured - 2e-7
    fitted = measured - 5e-8
    initial_rf = np.linspace(-3e-5, 3e-5, 6)
    fitted_rf = np.linspace(-2e-5, 2e-5, 6)
    measurement = tmp_path / "orm.h5"
    with h5py.File(measurement, "w") as handle:
        handle["response_matrix"] = measured
    with h5py.File(tmp_path / "model_orm_initial.h5", "w") as handle:
        handle["response_matrix"] = np.column_stack((initial, initial_rf))
    np.savez_compressed(tmp_path / "loco_results.npz", orm_model=np.column_stack((fitted, fitted_rf)))
    (tmp_path / "run_request.json").write_text(json.dumps({
        "measurements": {"orm": str(measurement)},
        "backend_mapping": {
            "MachineElements": {"bpm_ords": [10, 20, 30], "horizontal_corrector_ords": [40, 50], "vertical_corrector_ords": [60, 70]},
            "BadBPMPositions": [],
            "ExcludedCorrectorPositions": {"horizontal": [], "vertical": []},
            "LOCOOptions": {"includeDispersion": True},
            "RMConfig": {"includeDispersion": True},
        },
    }), encoding="utf-8")

    loader = ResultsLoader(tmp_path)
    assert loader.measured_orm.shape == loader.initial_orm.shape == loader.fitted_orm.shape == (6, 4)
    np.testing.assert_allclose(loader.initial_rf_response, initial_rf)
    np.testing.assert_allclose(loader.fitted_rf_response, fitted_rf)
    np.testing.assert_allclose(loader.residual_before, measured - initial)
    np.testing.assert_allclose(loader.residual_after, measured - fitted)
    assert loader.orm_rms_before == pytest.approx(np.sqrt(np.mean((measured - initial) ** 2)))
    assert loader.orm_rms_after == pytest.approx(np.sqrt(np.mean((measured - fitted) ** 2)))


def test_results_does_not_trim_unverified_extra_model_column(tmp_path):
    measurement = tmp_path / "orm.h5"
    with h5py.File(measurement, "w") as handle:
        handle["response_matrix"] = np.zeros((4, 2))
    with h5py.File(tmp_path / "model_orm_initial.h5", "w") as handle:
        handle["response_matrix"] = np.zeros((4, 3))
    np.savez_compressed(tmp_path / "loco_results.npz", orm_model=np.zeros((4, 3)))
    (tmp_path / "run_request.json").write_text(json.dumps({
        "measurements": {"orm": str(measurement)},
        "backend_mapping": {
            "MachineElements": {"bpm_ords": [10, 20], "horizontal_corrector_ords": [30], "vertical_corrector_ords": [40]},
            "BadBPMPositions": [],
            "ExcludedCorrectorPositions": {"horizontal": [], "vertical": []},
            "LOCOOptions": {"includeDispersion": False},
        },
    }), encoding="utf-8")
    loader = ResultsLoader(tmp_path)
    assert loader.initial_orm is None
    assert loader.fitted_orm is None
    assert "requires 2 corrector columns" in loader.unavailable_reason("initial_orm")


def test_restored_results_reproduce_optics_plots_and_statistics(app, tmp_path):
    _request(tmp_path / "run_request.json", included=True)
    _persisted(tmp_path, included=True)
    first = ResultsWorkspace(); first.load_results(tmp_path)
    second = ResultsWorkspace(); second.load_results(tmp_path)
    app.processEvents()
    for plane in ("x", "y"):
        np.testing.assert_array_equal(
            first.loader.beta_beating_data[f"beta_beating_{plane}"],
            second.loader.beta_beating_data[f"beta_beating_{plane}"],
        )
        assert first.loader.dispersion_statistics[plane] == second.loader.dispersion_statistics[plane]
        assert first.loader.beta_beating_statistics[plane] == second.loader.beta_beating_statistics[plane]
    first.close(); second.close()


def test_new_optics_artifact_persists_independent_dispersion_diagnostic(tmp_path):
    class Ring:
        def __init__(self, dispersion): self._dispersion = np.asarray(dispersion)
        def __len__(self): return 3
        def get_s_pos(self, refpts): return np.asarray(refpts, dtype=float) * 10
        def get_optics(self, refpts):
            indices = np.asarray(refpts, dtype=int)
            data = SimpleNamespace(beta=np.ones((len(indices), 2)), dispersion=self._dispersion[indices])
            return None, None, data
    reference = Ring([[1, 0, .1, 0], [2, 0, .2, 0], [3, 0, .3, 0]])
    fitted = Ring([[1.1, 0, .15, 0], [2.1, 0, .25, 0], [3.1, 0, .35, 0]])
    path = _save_optics_results(
        tmp_path, reference_ring=reference, fitted_ring=fitted,
        measured={"eta_x": np.array([.01, .02]), "eta_y": np.array([.001, .002]), "dispersion_supplied": True},
        initial_orm_path=tmp_path / "unused.h5", fitted_orm=None,
        include_dispersion=False, reference_kind="run_input_lattice", bpm_ords=[0, 2],
        rf_step=-10.0, rf_frequency=100.0, momentum_compaction=.01,
    )
    assert path == tmp_path / "optics_results.npz"
    _request(tmp_path / "run_request.json", included=False)
    reopened = ResultsLoader(tmp_path)
    assert reopened.dispersion_data is not None
    np.testing.assert_allclose(reopened.dispersion_data["x"]["measured"], [.01, .02])
    np.testing.assert_allclose(reopened.dispersion_data["x"]["initial"], [10, 30])
    assert reopened.dispersion_data["display_quantity"] == "rf_orbit_response"
