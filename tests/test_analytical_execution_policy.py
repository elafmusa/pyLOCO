import inspect

from pyLOCO.pyloco import compute_jacobian, pyloco


def test_new_execution_policy_defaults_preserve_legacy_fallbacks():
    for function in (compute_jacobian, pyloco):
        parameters = inspect.signature(function).parameters
        assert parameters["analytical_formula_use_mp"].default is None
        assert parameters["analytical_formula_workers"].default is None
        assert parameters["analytical_dispersion_use_mp"].default is None
        assert parameters["analytical_dispersion_workers"].default is None
        assert parameters["analytical_dispersion_worker"].default == "legacy_full_orm"
        assert parameters["skew_analytical_formula_use_mp"].default is None
        assert parameters["skew_analytical_formula_workers"].default is None
        assert parameters["skew_analytical_dispersion_use_mp"].default is None
        assert parameters["skew_analytical_dispersion_workers"].default is None


def test_petra_benchmarks_expose_independent_formula_and_dispersion_workers():
    from Examples.PETRAIV import benchmark_analytical_jacobian as normal
    from Examples.PETRAIV import benchmark_analytical_skew_jacobian as skew

    normal_args = normal._parse_args([
        "--formula-workers", "0", "--dispersion-workers", "64",
        "--dispersion-worker", "rf_only",
    ])
    assert normal_args.formula_workers == 0
    assert normal_args.dispersion_workers == 64
    assert normal_args.dispersion_worker == "rf_only"

    skew_args = skew._parse_args([
        "--formula-workers", "0", "--dispersion-workers", "32",
        "--dispersion-worker", "rf_only",
    ])
    assert skew_args.formula_workers == 0
    assert skew_args.dispersion_workers == 32
    assert skew_args.dispersion_worker == "rf_only"

