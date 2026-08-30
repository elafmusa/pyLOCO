import numpy as np
import pytest

from pyLOCO.pyloco import _require_finite_evaluation, solve_step_lm


def test_nonfinite_diagnostic_reports_full_numerical_context(capsys):
    with pytest.raises(FloatingPointError) as excinfo:
        _require_finite_evaluation(
            np.asarray([[1.0, np.nan], [np.inf, 2.0]]),
            evaluation="central_difference_minus_orm",
            calculator="Tracking",
            iteration=3,
            block="skew_quads",
            group=[64, 82],
            step=-2.5e-2,
        )

    message = str(excinfo.value)
    assert "evaluation=central_difference_minus_orm" in message
    assert "calculator=Tracking" in message
    assert "iteration=3" in message
    assert "block=skew_quads" in message
    assert "group=[64, 82]" in message
    assert "step=-2.500000000000e-02" in message
    assert "nonfinite_count=2" in message
    assert "NON-FINITE LOCO EVALUATION" in capsys.readouterr().out


def test_nonfinite_trial_diagnostic_can_report_without_raising():
    count = _require_finite_evaluation(
        np.asarray([1.0, np.nan, np.inf]),
        evaluation="lm_trial_orm_inner_2",
        calculator="Tracking",
        iteration=1,
        raise_on_nonfinite=False,
    )
    assert count == 2


def test_lm_solver_rejects_nonfinite_jacobian_before_svd():
    with pytest.raises(FloatingPointError, match="lm_solver_weighted_jacobian"):
        solve_step_lm(
            np.asarray([[1.0, np.nan], [0.0, 1.0]]),
            np.ones((2, 1)),
            np.ones(2),
            np.zeros((2, 1)),
            np.zeros((2, 1)),
            tag="LM it2/in1",
        )
