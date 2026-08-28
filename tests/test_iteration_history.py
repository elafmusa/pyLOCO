import json
import sys
import types
from pathlib import Path

import numpy as np

from pyLOCO.gui.backend import LocoRunRequest, _save_iteration_snapshot, _write_iteration_manifest
from pyLOCO.gui.results.results_loader import ResultsLoader


class _Optics:
    def __init__(self, scale):
        self.beta = np.asarray([[1.0, 2.0], [2.0, 3.0]]) * scale
        self.dispersion = np.asarray([[0.1, 0.0, 0.2, 0.0], [0.2, 0.0, 0.3, 0.0]]) * scale


class _Ring:
    def __init__(self, scale):
        self.scale = scale

    def __len__(self):
        return 2

    def get_optics(self, refpts):
        return None, None, _Optics(self.scale)

    def get_s_pos(self, refpts):
        return np.asarray(refpts, dtype=float)


def _request(tmp_path):
    return LocoRunRequest(
        project_name="tiny", project_path=str(tmp_path / "tiny.pyloco.json"),
        lattice_path=str(tmp_path / "ring.mat"), measurements={},
        backend_mapping={
            "LOCOOptions": {"algorithm": "lm", "quad_jacobian_calculator": "Numerical",
                            "skew_jacobian_calculator": "Numerical"},
            "RMConfig": {"calculator": "Linear"},
        },
    )


def _record(iteration, vector, ring_scale):
    return {
        "iteration": iteration, "chi2_before": 10.0 / (iteration + 1),
        "chi2_after": 10.0 / (iteration + 1), "fit_parameters": np.asarray(vector),
        "blocks": {"quads": slice(0, 2)}, "ring": _Ring(ring_scale),
        "orm_model": np.full((4, 2), ring_scale), "timings": {"iteration_seconds": iteration},
        "orm_residual": {"rms": ring_scale},
    }


def test_iteration_snapshots_are_atomic_ordered_and_reopenable(tmp_path, monkeypatch):
    fake_at = types.SimpleNamespace(save_lattice=lambda ring, path: Path(path).write_text(str(ring.scale)))
    monkeypatch.setitem(sys.modules, "at", fake_at)
    request = _request(tmp_path)
    reference = _Ring(1.0)
    measured = {"dispersion_supplied": False, "eta_x": None, "eta_y": None}
    for number, vector in enumerate(([1.0, 2.0], [1.1, 2.2], [1.3, 2.5])):
        record = _record(number, vector, 1.0 + number / 10)
        _save_iteration_snapshot(
            tmp_path, record, diagnostics={"iteration": number, "chi2_before": record["chi2_before"],
                                            "chi2_after": record["chi2_after"]},
            reference_ring=reference, measured=measured, include_dispersion=False,
            bpm_ords=[0, 1], rf_step=1.0, rf_frequency=1.0,
            momentum_compaction=1.0, request=request,
        )
    _write_iteration_manifest(tmp_path, run_status="completed")
    (tmp_path / "run_request.json").write_text(json.dumps({"backend_mapping": request.backend_mapping}))
    (tmp_path / "summary.json").write_text(json.dumps({"initial_chi2": 10.0,
                                                         "chi2_history": [5.0, 2.0],
                                                         "blocks": {"quads": {"start": 0, "stop": 2}}}))

    loader = ResultsLoader(tmp_path)
    assert [entry["iteration"] for entry in loader.iteration_entries] == [0, 1, 2]
    assert loader.iteration_entries[-1]["label"] == "Iteration 2 / Final"
    np.testing.assert_allclose(loader.for_iteration(2).parameter_vector, [1.3, 2.5])
    np.testing.assert_allclose(loader.for_iteration(2).cumulative_parameter_change, [0.3, 0.5])
    np.testing.assert_allclose(loader.for_iteration(2).iteration_parameter_step, [0.2, 0.3])
    np.testing.assert_allclose(loader.for_iteration(2).fitted_orm, np.full((4, 2), 1.2))
    assert not list((tmp_path / "iterations").glob(".iteration_*"))


def test_failed_manifest_preserves_only_completed_snapshots(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "at", types.SimpleNamespace(
        save_lattice=lambda ring, path: Path(path).write_text(str(ring.scale))))
    record = _record(0, [1.0, 2.0], 1.0)
    _save_iteration_snapshot(
        tmp_path, record, diagnostics={"iteration": 0, "chi2_after": 10.0},
        reference_ring=_Ring(1.0), measured={"dispersion_supplied": False},
        include_dispersion=False, bpm_ords=[0, 1], rf_step=1.0,
        rf_frequency=1.0, momentum_compaction=1.0, request=_request(tmp_path))
    (tmp_path / "iterations" / ".iteration_001-incomplete").mkdir()
    manifest = _write_iteration_manifest(tmp_path, run_status="failed")
    data = json.loads(manifest.read_text())
    assert data["run_status"] == "failed"
    assert [item["iteration"] for item in data["iterations"]] == [0]


def test_old_final_only_results_remain_loadable(tmp_path):
    np.savez_compressed(tmp_path / "loco_results.npz", fit_results=np.asarray([[1.0, 2.0]]),
                        orm_model=np.zeros((2, 2)), chi2_history=np.asarray([1.0]))
    loader = ResultsLoader(tmp_path)
    assert loader.iteration_entries == [{"iteration": None, "label": "Final", "legacy": True}]
    np.testing.assert_allclose(loader.parameter_vector, [1.0, 2.0])


def test_results_workspace_selector_switches_all_views(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "at", types.SimpleNamespace(
        save_lattice=lambda ring, path: Path(path).write_text(str(ring.scale))))
    request = _request(tmp_path)
    for number, vector in enumerate(([1.0, 2.0], [1.2, 2.4])):
        record = _record(number, vector, 1.0 + number / 10)
        _save_iteration_snapshot(
            tmp_path, record, diagnostics={"iteration": number, "chi2_before": 10.0,
                                            "chi2_after": 10.0 / (number + 1)},
            reference_ring=_Ring(1.0), measured={"dispersion_supplied": False},
            include_dispersion=False, bpm_ords=[0, 1], rf_step=1.0,
            rf_frequency=1.0, momentum_compaction=1.0, request=request)
    _write_iteration_manifest(tmp_path, run_status="completed")
    (tmp_path / "run_request.json").write_text(json.dumps({"backend_mapping": request.backend_mapping}))
    (tmp_path / "summary.json").write_text(json.dumps({"initial_chi2": 10.0,
                                                         "chi2_history": [5.0],
                                                         "blocks": {"quads": {"start": 0, "stop": 2}}}))
    from pyLOCO.gui.results.results_workspace import ResultsWorkspace
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    workspace = ResultsWorkspace()
    workspace.load_results(tmp_path)
    assert workspace.iteration_selector.count() == 2
    assert workspace.iteration_selector.currentText() == "Iteration 1 / Final"
    assert workspace.loader.iteration == 1
    workspace.iteration_selector.setCurrentIndex(0)
    assert workspace.loader.iteration == 0
    assert workspace.overview.loader is workspace.loader
    workspace.close()
