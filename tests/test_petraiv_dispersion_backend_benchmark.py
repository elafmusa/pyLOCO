import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "Examples" / "PETRAIV" / "benchmark_dispersion_derivative_backends.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("dispersion_backend_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plane_metrics_preserve_undefined_relative_norm_for_zero_reference():
    metrics = _module()._plane_metrics(np.zeros((2, 3)), np.zeros((2, 3)))
    assert metrics["relative_norm_difference"] is None
    assert metrics["reference_finite_count"] == metrics["element_count"] == 6


def test_missing_input_reports_resolved_path(tmp_path):
    missing = tmp_path / "missing.mat"
    with pytest.raises(FileNotFoundError, match=str(missing.resolve())):
        _module()._resolved_file(missing, "Lattice")
