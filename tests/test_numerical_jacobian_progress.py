from __future__ import annotations

import numpy as np

import pyLOCO.pyloco as pyloco
from pyLOCO.config import FitInitConfig


def test_single_worker_numerical_jacobian_reports_each_parameter(monkeypatch, tmp_path):
    calls = []

    def evaluate(quad_index, *_args, **_kwargs):
        return np.full((2, 2), float(quad_index)), np.array([1.0e-6]), []

    monkeypatch.setattr(pyloco, "available_worker_count", lambda **_kwargs: 1)
    monkeypatch.setattr(pyloco, "generating_quads_response_matrices", evaluate)

    class ImmediateResult:
        def __init__(self, function, args):
            self.value = function(*args)

        def get(self, timeout=None):
            return self.value

    class ImmediatePool:
        def __init__(self, *, initializer, initargs, **_kwargs):
            initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def apply_async(self, function, args):
            return ImmediateResult(function, args)

    class ImmediateContext:
        Pool = ImmediatePool

    monkeypatch.setattr(pyloco.mp, "get_context", lambda _method: ImmediateContext())

    jacobian, steps = pyloco.calculate_quads_jacobian(
        ring=object(),
        C_model=np.zeros((2, 2)),
        dkick=(np.array([1.0e-4]), np.array([1.0e-4])),
        used_cor_ind=(np.array([0]), np.array([1])),
        bpm_indexes=np.array([0]),
        quads_ind=np.array([3, 7]),
        dk=1.0e-6,
        C=np.zeros((2, 2)),
        individuals=True,
        HCMCoupling=np.array([0.0]),
        VCMCoupling=np.array([0.0]),
        rf_step=200.0,
        block="quads",
        fit_cfg=FitInitConfig(),
        output_dir=tmp_path,
        progress_callback=lambda block, done, total: calls.append((block, done, total)),
    )

    assert jacobian.shape == (2, 2, 2)
    assert steps.shape == (2,)
    assert calls == [("quads", 1, 2), ("quads", 2, 2)]
