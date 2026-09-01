import inspect
import importlib

import pytest

from pyLOCO.pyloco import compute_jacobian, pyloco

pyloco_module = importlib.import_module("pyLOCO.pyloco")


def test_dispersion_difference_validation():
    assert pyloco_module._normalize_dispersion_difference("Central") == "central"
    assert pyloco_module._normalize_dispersion_difference("forward") == "forward"
    with pytest.raises(ValueError, match="central.*forward"):
        pyloco_module._normalize_dispersion_difference("backward")


def test_dispersion_step_metric_validation():
    assert pyloco_module._normalize_dispersion_step_metric("Full_ORM") == "full_orm"
    assert pyloco_module._normalize_dispersion_step_metric("rf_only") == "rf_only"
    with pytest.raises(ValueError, match="full_orm.*rf_only"):
        pyloco_module._normalize_dispersion_step_metric("orm_columns")


def test_new_execution_policy_defaults_preserve_legacy_fallbacks():
    for function in (compute_jacobian, pyloco):
        parameters = inspect.signature(function).parameters
        assert parameters["analytical_formula_use_mp"].default is None
        assert parameters["analytical_formula_workers"].default is None
        assert parameters["analytical_dispersion_use_mp"].default is None
        assert parameters["analytical_dispersion_workers"].default is None
        assert parameters["analytical_dispersion_worker"].default == "legacy_full_orm"
        assert parameters["analytical_dispersion_difference"].default == "central"
        assert parameters["analytical_dispersion_step_metric"].default == "full_orm"
        assert parameters["analytical_dispersion_worker_transport"].default == "per_task"
        assert parameters["analytical_dispersion_worker_chunksize"].default == 1
        assert parameters["analytical_dispersion_reuse_adaptive_plus_rf"].default is False
        assert parameters["skew_analytical_formula_use_mp"].default is None
        assert parameters["skew_analytical_formula_workers"].default is None
        assert parameters["skew_analytical_dispersion_use_mp"].default is None
        assert parameters["skew_analytical_dispersion_workers"].default is None
        assert parameters["skew_analytical_dispersion_difference"].default == "central"
        assert parameters["skew_analytical_dispersion_step_metric"].default == "full_orm"
        assert parameters["skew_analytical_dispersion_worker_transport"].default == "per_task"
        assert parameters["skew_analytical_dispersion_worker_chunksize"].default == 1
        assert parameters["skew_analytical_dispersion_reuse_adaptive_plus_rf"].default is False


def test_petra_benchmarks_expose_independent_formula_and_dispersion_workers():
    from Examples.PETRAIV import benchmark_analytical_jacobian as normal
    from Examples.PETRAIV import benchmark_analytical_skew_jacobian as skew

    normal_args = normal._parse_args([
        "--formula-workers", "0", "--dispersion-workers", "64",
        "--dispersion-worker", "rf_only",
        "--dispersion-difference", "forward",
        "--dispersion-step-metric", "rf_only",
        "--dispersion-worker-transport", "initializer",
        "--dispersion-chunksize", "4", "--reuse-adaptive-plus-rf",
    ])
    assert normal_args.formula_workers == 0
    assert normal_args.dispersion_workers == 64
    assert normal_args.dispersion_worker == "rf_only"
    assert normal_args.dispersion_difference == "forward"
    assert normal_args.dispersion_step_metric == "rf_only"
    assert normal_args.dispersion_worker_transport == "initializer"
    assert normal_args.dispersion_chunksize == 4
    assert normal_args.reuse_adaptive_plus_rf is True

    skew_args = skew._parse_args([
        "--formula-workers", "0", "--dispersion-workers", "32",
        "--dispersion-worker", "rf_only",
        "--dispersion-difference", "forward",
        "--dispersion-step-metric", "rf_only",
        "--dispersion-worker-transport", "initializer",
        "--dispersion-chunksize", "4", "--reuse-adaptive-plus-rf",
    ])
    assert skew_args.formula_workers == 0
    assert skew_args.dispersion_workers == 32
    assert skew_args.dispersion_worker == "rf_only"
    assert skew_args.dispersion_difference == "forward"
    assert skew_args.dispersion_step_metric == "rf_only"
    assert skew_args.dispersion_worker_transport == "initializer"
    assert skew_args.dispersion_chunksize == 4
    assert skew_args.reuse_adaptive_plus_rf is True
