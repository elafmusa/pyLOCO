from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
import pytest


from pyLOCO.measured_machine.diagnostics import extract_corrections, safe_relative_percent
from pyLOCO.measured_machine.workflow import (
    _load_family_groups,
    _require_datasets,
    build_constraint_config,
    load_resume_state,
    output_directory,
)


class Element:
    def __init__(self, normal: float, skew: float = 0.0):
        self.K = normal
        self.PolynomB = np.asarray([0.0, normal])
        self.PolynomA = np.asarray([0.0, skew])
        self.FamName = "Q"


def test_combined_dispersion_dataset_validation(tmp_path):
    path = tmp_path / "dispersion.h5"
    with h5py.File(path, "w") as stream:
        stream["measured_eta_x"] = [1.0]
        with pytest.raises(ValueError, match="measured_eta_y"):
            _require_datasets(stream, path, ("measured_eta_x", "measured_eta_y"))


def test_family_group_loading(tmp_path):
    path = tmp_path / "families.npy"
    np.save(path, np.asarray([np.asarray([4, 7]), np.asarray([9])], dtype=object), allow_pickle=True)
    assert _load_family_groups(path) == [[4, 7], [9]]


def test_constraint_config_relative_sigma_and_weights():
    ring = [Element(2.0), Element(-3.0), Element(0.0, 0.2)]
    data = {
        "ring": ring, "quad_indices": [[0], [1]], "skew_indices": np.asarray([2]),
        "cfg": {"constraints": {"enable": True, "quadrupoles": {
            "relative_sigma": 1.0e-4, "default_weight": 1.0,
            "weighted_families": {1: 5.0}}, "skew_quadrupoles": {
            "sigma": 1.0e-4, "default_weight": 0.0}}},
    }
    constraint = build_constraint_config(data)
    np.testing.assert_allclose(constraint.quad_sigma, [2.0e-4, 3.0e-4])
    np.testing.assert_allclose(constraint.quad_weights, [1.0, 5.0])
    np.testing.assert_allclose(constraint.skew_weights, [0.0])


def test_constraint_rejects_invalid_weighted_family():
    data = {"ring": [Element(1.0)], "quad_indices": [[0]], "skew_indices": np.asarray([]),
            "cfg": {"constraints": {"enable": True, "quadrupoles": {
                "weighted_families": {2: 5.0}}}}}
    with pytest.raises(ValueError, match="outside"):
        build_constraint_config(data)


def test_constraint_selected_families_override_default_weight():
    data = {"ring": [Element(1.0), Element(2.0), Element(3.0)],
            "quad_indices": [[0], [1], [2]], "skew_indices": np.asarray([]),
            "cfg": {"constraints": {"enable": True, "quadrupoles": {
                "default_weight": 0.0, "selected_weight": 5.0,
                "selected_families": [0, 2]}}}}
    constraint = build_constraint_config(data)
    np.testing.assert_allclose(constraint.quad_weights, [5.0, 0.0, 5.0])


def test_constraint_rejects_invalid_selected_family():
    data = {"ring": [Element(1.0)], "quad_indices": [[0]],
            "skew_indices": np.asarray([]), "cfg": {"constraints": {
                "enable": True, "quadrupoles": {"selected_weight": 5.0,
                    "selected_families": [1]}}}}
    with pytest.raises(ValueError, match="outside"):
        build_constraint_config(data)


def test_family_correction_sign_and_expansion():
    initial = [Element(2.0), Element(2.0), Element(-4.0), Element(0.0, 0.1)]
    fitted = [Element(2.2), Element(2.2), Element(-3.6), Element(0.0, 0.08)]
    data = {"ring": initial, "quad_indices": [[0, 1], [2]], "skew_indices": np.asarray([3]),
            "cfg": {"loco": {"skew_attribute": "PolynomA"}}}
    fit = {"ring": fitted, "fit_list": ["quads", "skew_quads"],
           "fit_dict": {0: {"quads": np.asarray([2.2, -3.6]), "skew_quads": np.asarray([0.08])}}}
    correction = extract_corrections(data, fit)
    np.testing.assert_allclose(correction["delta_q_families"], [-0.2, -0.4])
    np.testing.assert_allclose(correction["delta_q_expanded"], [-0.2, -0.2, -0.4])
    np.testing.assert_allclose(correction["expanded_indices"], [0, 1, 2])
    np.testing.assert_allclose(correction["delta_skew"], [0.02])


def test_zero_strength_relative_correction_is_nan():
    assert np.isnan(safe_relative_percent([1.0], [0.0])[0])


def test_output_directory_is_relative_to_configuration(tmp_path):
    config = tmp_path / "configs" / "case.yaml"
    data = {"config_path": config, "cfg": {"output": {"directory": "../output/case"}}}
    assert output_directory(data, coupling=False, constrained=False) == (tmp_path / "output" / "case").resolve()


def test_output_root_uses_run_name(tmp_path):
    config = tmp_path / "configs" / "case.yaml"
    data = {"config_path": config, "cfg": {"output": {"root": "../output", "run_name": "after_fit"}}}
    assert output_directory(data, coupling=True, constrained=True) == (tmp_path / "output" / "after_fit").resolve()


def test_resume_is_optional(tmp_path):
    assert load_resume_state(None, tmp_path) is None
    assert load_resume_state({"enabled": False}, tmp_path) is None


def test_resume_reports_missing_saved_state(tmp_path):
    with pytest.raises(FileNotFoundError, match="ring_pyloco.mat.*fit_dict.pkl"):
        load_resume_state({"enabled": True, "directory": "first_stage"}, tmp_path)
