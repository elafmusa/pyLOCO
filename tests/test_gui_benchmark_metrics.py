import csv
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pyLOCO.gui.backend import _iteration_diagnostics, _save_iteration_metrics_csv, _save_outputs
from pyLOCO.gui.results.results_loader import ResultsLoader
from pyLOCO.gui.models.project import LocoConfiguration
from pyLOCO.pyloco import _calculator_execution_plan


@pytest.mark.parametrize(("orm", "jacobian"), [
    ("Linear", "Numerical"), ("Analytical", "Numerical"),
    ("Numerical", "Numerical"), ("Linear", "Analytical"),
    ("Analytical", "Analytical"), ("Numerical", "Analytical"),
])
def test_compute_jacobian_instrumentation_for_all_six_combinations(
    tmp_path, monkeypatch, orm, jacobian
):
    module = importlib.import_module("pyLOCO.pyloco")
    calls = {"orm": [], "numerical": [], "analytical": 0}
    marker = {"Linear": 1.0, "Analytical": 2.0, "Numerical": 3.0}[orm]

    def response_matrix(_ring, *, config):
        calls["orm"].append(config.calculator)
        return np.full((2, 2), marker)

    def numerical(_ring, model, *args, orm_calculator=None, **kwargs):
        calls["numerical"].append((orm_calculator, np.asarray(model).copy()))
        return np.zeros((1, 2, 2)), np.asarray([1e-4])

    def analytical(**kwargs):
        calls["analytical"] += 1
        return np.zeros((1, 2, 2)), None

    monkeypatch.setattr(module, "response_matrix", response_matrix)
    monkeypatch.setattr(module, "calculate_quads_jacobian", numerical)
    monkeypatch.setattr(module, "calculate_quads_jacobian_analytical", analytical)
    trace = []
    model = response_matrix(None, config=SimpleNamespace(calculator=orm))
    result, *_ = module.compute_jacobian(
        ring=[], C_model=model, dkick=[[1e-5], [1e-5]], dk=1e-4,
        bpm_indexes=np.asarray([0]), CMords=[np.asarray([0]), np.asarray([1])],
        quads_ind=np.asarray([0]), nHorCOR=1, nVerCOR=1, nHBPM=1, nVBPM=1,
        C=np.eye(2), CAVords=np.asarray([], dtype=int), quad_individuals=True,
        HCMCoupling=np.zeros(1), VCMCoupling=np.zeros(1), fit_cfg=SimpleNamespace(),
        response_matrix_calculator=orm, quad_jacobian_calculator=jacobian,
        calculator_trace_callback=trace.append, output_dir=tmp_path / f"{orm}-{jacobian}",
    )
    assert result.shape == (1, 2, 2)
    assert calls["orm"][0] == orm  # selected calculator for the main model
    if jacobian == "Numerical":
        assert calls["numerical"][0][0] == orm
        assert np.all(calls["numerical"][0][1] == marker)
        assert calls["analytical"] == 0
        assert trace[-1] == {
            "stage": "normal_quad_numerical_perturbation_orm", "calculator": orm
        }
    else:
        assert calls["numerical"] == []
        assert calls["analytical"] == 1
        assert trace[-1] == {
            "stage": "normal_quad_analytical_derivative", "calculator": "Analytical"
        }


@pytest.mark.parametrize(("orm", "backend_orm", "jacobian"), [
    ("Linear", "Linear", "Numerical"),
    ("Analytical", "Analytical", "Numerical"),
    ("Tracking", "Numerical", "Numerical"),
    ("Linear", "Linear", "Analytical"),
    ("Analytical", "Analytical", "Analytical"),
    ("Tracking", "Numerical", "Analytical"),
])
def test_all_orm_and_normal_jacobian_combinations_are_independent(
    orm, backend_orm, jacobian
):
    config = LocoConfiguration()
    config.response_matrix.calculator = orm
    config.rejection.quad_jacobian_calculator = jacobian
    mapping = config.to_backend_mapping()
    plan = _calculator_execution_plan(
        mapping["RMConfig"]["calculator"],
        mapping["LOCOOptions"]["quad_jacobian_calculator"],
    )

    assert plan["response_matrix_calculator"] == backend_orm
    assert plan["normal_quad_jacobian"] == jacobian
    if jacobian == "Numerical":
        assert plan["numerical_jacobian_orm_calculator"] == backend_orm
    else:
        assert plan["numerical_jacobian_orm_calculator"] is None

    # Changing one selector never rewrites the other project setting.
    original_jacobian = config.rejection.quad_jacobian_calculator
    config.response_matrix.calculator = "Linear" if orm != "Linear" else "Tracking"
    assert config.rejection.quad_jacobian_calculator == original_jacobian
    original_orm = config.response_matrix.calculator
    config.rejection.quad_jacobian_calculator = (
        "Analytical" if jacobian == "Numerical" else "Numerical"
    )
    assert config.response_matrix.calculator == original_orm


class _Ring:
    def __init__(self, beta, dispersion):
        self._beta = np.asarray(beta, dtype=float)
        self._dispersion = np.asarray(dispersion, dtype=float)

    def __len__(self):
        return len(self._beta)

    def get_optics(self, refpts):
        indices = np.asarray(refpts, dtype=int)
        data = SimpleNamespace(
            beta=self._beta[indices],
            dispersion=self._dispersion[indices],
        )
        return None, None, data


def test_iteration_diagnostics_records_optics_and_dispersion_metrics():
    reference = _Ring(
        beta=[[10.0, 20.0], [20.0, 40.0]],
        dispersion=[[1.0, 0.0, 2.0, 0.0], [2.0, 0.0, 4.0, 0.0]],
    )
    fitted = _Ring(
        beta=[[11.0, 18.0], [22.0, 36.0]],
        dispersion=[[0.9, 0.0, 2.1, 0.0], [1.9, 0.0, 4.1, 0.0]],
    )
    # conversion = -alpha_c * f / df = 1 for these values
    result = _iteration_diagnostics(
        {"iteration": 1, "ring": fitted, "orm_model": np.zeros((4, 2)),
         "timings": {"jacobian_seconds": 2.5}},
        reference_ring=reference,
        measured={"dispersion_supplied": True,
                  "eta_x": np.asarray([1.0, 2.0]),
                  "eta_y": np.asarray([2.0, 4.0])},
        bpm_ords=np.asarray([0, 1]), rf_step=-1.0,
        rf_frequency=1.0, momentum_compaction=1.0,
    )

    assert result["beta_beating_percent"]["x"]["rms"] == 10.0
    assert result["beta_beating_percent"]["y"]["rms"] == 10.0
    assert np.isclose(result["dispersion_residual_m"]["x"]["rms"], 0.1)
    assert np.isclose(result["dispersion_residual_m"]["y"]["rms"], 0.1)
    assert "ring" not in result
    assert "orm_model" not in result


def test_iteration_metrics_csv_and_loader_are_backward_compatible(tmp_path):
    record = {
        "iteration": 1, "chi2_before": 12.0, "chi2_after": 4.0,
        "orm_residual": {"rms": 2e-6},
        "horizontal_orm_residual": {"rms": 1e-6},
        "vertical_orm_residual": {"rms": 3e-6},
        "beta_beating_percent": {"x": {"rms": 1.2}, "y": {"rms": 2.3}},
        "dispersion_residual_m": {"x": {"rms": 0.004}, "y": {"rms": 0.005}},
        "timings": {"model_orm_seconds": 1.0, "jacobian_seconds": 2.0,
                    "trial_orm_seconds": 3.0, "final_orm_seconds": 4.0,
                    "total_orm_seconds": 8.0, "iteration_seconds": 9.0},
    }
    (tmp_path / "summary.json").write_text(
        json.dumps({"iteration_metrics": [record]}), encoding="utf-8"
    )
    _save_iteration_metrics_csv(tmp_path / "iteration_metrics.csv", [record])

    loader = ResultsLoader(tmp_path)
    assert loader.iteration_metrics[0]["chi2_after"] == 4.0
    assert loader.timing_totals["jacobian_seconds"] == 2.0
    with (tmp_path / "iteration_metrics.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["dispersion_x_rms_mm"] == "4.0"
    assert row["iteration_seconds"] == "9.0"

    empty = tmp_path / "legacy"
    empty.mkdir()
    assert ResultsLoader(empty).iteration_metrics == []
    assert ResultsLoader(empty).timing_totals["jacobian_seconds"] is None


def test_saved_summary_records_independent_calculator_metadata(tmp_path):
    def save_fit_dict(_value, path):
        path.write_text("{}", encoding="utf-8")

    _save_outputs(
        tmp_path, [np.asarray([1.0])], {}, object(), np.zeros((2, 2)),
        np.eye(2), [4.0], [], {"quads": slice(0, 1)}, save_fit_dict,
        response_matrix_calculator="Tracking",
        response_matrix_backend_calculator="Numerical",
        normal_quad_jacobian="Numerical",
        analytical_implementation="legacy",
        calculator_trace=[{
            "stage": "normal_quad_numerical_perturbation_orm",
            "calculator": "Numerical",
        }],
    )
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["response_matrix_calculator"] == "Tracking"
    assert summary["response_matrix_backend_calculator"] == "Numerical"
    assert summary["normal_quad_jacobian"] == "Numerical"
    assert summary["analytical_implementation"] == "legacy"
    assert summary["normal_quad_jacobian_orm_calculator"] == "Tracking"
    assert summary["calculator_trace"][0]["calculator"] == "Numerical"
