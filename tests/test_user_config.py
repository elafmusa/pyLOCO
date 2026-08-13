import numpy as np

from pyLOCO.user_config import build_constraints, selected_fit_parameters


def _config(enable_constraints=False, weight=5.0):
    return {
        "fit_parameters": {
            "quadrupoles": {"enable": True},
            "skew_quadrupoles": {"enable": True},
        },
        "constraints": {
            "enable": enable_constraints,
            "quadrupoles": {
                "relative_sigma": 1e-4, "minimum_sigma": 1e-12,
                "default_weight": 0.0, "selected_weight": weight,
                "selected_families": [0, 2],
            },
            "skew_quadrupoles": {"sigma": 1e-4, "default_weight": 0.0},
        },
    }


def test_quadrupoles_are_selected_when_constraints_are_off():
    cfg = _config(False)
    assert "quads" in selected_fit_parameters(cfg)
    assert build_constraints(cfg, quad_nominal=[1, 2, 3], n_skew=1) is None


def test_selected_quadrupole_families_receive_common_weight():
    constraints = build_constraints(_config(True, 5), quad_nominal=[1, 2, 3], n_skew=1)
    np.testing.assert_allclose(constraints.quad_weights, [5, 0, 5])
    np.testing.assert_allclose(constraints.quad_sigma, [1e-4, 2e-4, 3e-4])


def test_changing_selected_weight_changes_only_selected_families():
    one = build_constraints(_config(True, 1), quad_nominal=[1, 2, 3], n_skew=1)
    five = build_constraints(_config(True, 5), quad_nominal=[1, 2, 3], n_skew=1)
    np.testing.assert_allclose(one.quad_weights, [1, 0, 1])
    np.testing.assert_allclose(five.quad_weights, [5, 0, 5])


def test_skew_selection_and_skew_constraint_weight_are_independent():
    cfg = _config(True)
    cfg["fit_parameters"]["skew_quadrupoles"]["enable"] = False
    cfg["constraints"]["skew_quadrupoles"]["default_weight"] = 3
    assert "skew_quads" not in selected_fit_parameters(cfg)
    constraints = build_constraints(cfg, quad_nominal=[1, 2, 3], n_skew=2)
    np.testing.assert_allclose(constraints.skew_weights, [3, 3])


def test_legacy_fit_list_remains_supported():
    assert selected_fit_parameters({"loco": {"fit_list": ["quads", "hbpm_gain"]}}) == [
        "hbpm_gain", "quads"
    ]
