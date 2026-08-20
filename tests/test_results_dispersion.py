import json

import h5py
import numpy as np
import pytest
from types import SimpleNamespace

from pyLOCO.gui.backend import _save_optics_results
from pyLOCO.gui.results.results_loader import ResultsLoader


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
    np.testing.assert_allclose(reopened.dispersion_data["x"]["measured"], [.001, .002])
    np.testing.assert_allclose(reopened.dispersion_data["x"]["initial"], [1, 3])
