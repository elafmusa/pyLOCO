import numpy as np
import at

import pyLOCO.analytic_orm_with_normal_quad_errors as normal_module
import pyLOCO.analytic_orm_with_skew_quad_errors as skew_module
import pyLOCO.pyloco as pyloco_module


class _SerialPool:
    """Exercise Pool argument packing deterministically in the test process."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def starmap_async(self, function, arguments):
        values = [function(*args) for args in arguments]
        return type("Ready", (), {"get": lambda self, timeout=None: values})()

    def terminate(self):
        pass


def _ring_and_indices():
    drift = at.Drift("D", 0.4)
    cell = at.Lattice([
        drift,
        at.Monitor("BPM"),
        at.Corrector("COR", 0.0, [0.0, 0.0]),
        at.Quadrupole("QF", 0.25, 0.9),
        drift,
        at.Quadrupole("QD", 0.25, -0.9),
    ], energy=1e9)
    ring = cell * 12
    bpms = np.asarray(at.get_refpts(ring, at.Monitor), dtype=int)[:3]
    correctors = np.asarray(at.get_refpts(ring, at.Corrector), dtype=int)[:3]
    quadrupoles = np.asarray(at.get_refpts(ring, at.Quadrupole), dtype=int)[:2]
    return ring, bpms, correctors, quadrupoles


def test_normal_multiprocessing_accepts_python_integer_ordinals(monkeypatch):
    ring, bpms, correctors, quadrupoles = _ring_and_indices()
    monkeypatch.setattr(normal_module.multiprocessing, "Pool", _SerialPool)
    expected = normal_module.analytic_orm_variation_with_normal_quadrupole(
        ring, bpms, correctors, quadrupoles, verbose=False,
        thick_quadrupole=True, thick_steerers=False, use_mp=False)
    actual = normal_module.analytic_orm_variation_with_normal_quadrupole(
        ring, bpms, correctors, quadrupoles, verbose=False,
        thick_quadrupole=True, thick_steerers=False, use_mp=True)
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])


def test_vectorized_normal_derivative_matches_legacy_thick_result():
    ring, bpms, correctors, quadrupoles = _ring_and_indices()
    legacy = normal_module.analytic_orm_variation_with_normal_quadrupole(
        ring, bpms, correctors, quadrupoles, verbose=False,
        thick_quadrupole=True, thick_steerers=False,
        implementation="legacy",
    )
    vectorized = normal_module.analytic_orm_variation_with_normal_quadrupole(
        ring, bpms, correctors, quadrupoles, verbose=False,
        thick_quadrupole=True, thick_steerers=False,
        implementation="vectorized",
    )
    np.testing.assert_allclose(vectorized[0], legacy[0], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(vectorized[1], legacy[1], rtol=2e-14, atol=2e-14)


def test_vectorized_normal_derivative_matches_legacy_thin_result():
    ring, bpms, correctors, quadrupoles = _ring_and_indices()
    legacy = normal_module.analytic_orm_variation_with_normal_quadrupole(
        ring, bpms, correctors, quadrupoles[:1], verbose=False,
        thick_quadrupole=False, thick_steerers=False,
        implementation="legacy",
    )
    vectorized = normal_module.analytic_orm_variation_with_normal_quadrupole(
        ring, bpms, correctors, quadrupoles[:1], verbose=False,
        thick_quadrupole=False, thick_steerers=False,
        implementation="vectorized",
    )
    np.testing.assert_allclose(vectorized[0], legacy[0], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(vectorized[1], legacy[1], rtol=2e-14, atol=2e-14)


def test_vectorized_subset_order_and_progress_are_preserved():
    ring, bpms, correctors, quadrupoles = _ring_and_indices()
    reversed_quads = quadrupoles[::-1]
    progress = []
    reversed_result = normal_module.analytic_orm_variation_with_normal_quadrupole(
        ring, bpms, correctors, reversed_quads, verbose=False,
        thick_quadrupole=True, thick_steerers=False,
        progress_callback=lambda done, total: progress.append((done, total)),
    )
    forward_result = normal_module.analytic_orm_variation_with_normal_quadrupole(
        ring, bpms, correctors, quadrupoles, verbose=False,
        thick_quadrupole=True, thick_steerers=False,
    )
    np.testing.assert_allclose(reversed_result[0], forward_result[0][:, :, ::-1])
    np.testing.assert_allclose(reversed_result[1], forward_result[1][:, :, ::-1])
    assert progress[-1] == (len(reversed_quads), len(reversed_quads))


def test_legacy_subset_order_and_both_planes_are_preserved():
    ring, bpms, correctors, quadrupoles = _ring_and_indices()
    subset = quadrupoles[::-1]
    legacy = normal_module.analytic_orm_variation_with_normal_quadrupole(
        ring, bpms, correctors, subset, verbose=False,
        thick_quadrupole=True, thick_steerers=False, implementation="legacy",
    )
    vectorized = normal_module.analytic_orm_variation_with_normal_quadrupole(
        ring, bpms, correctors, subset, verbose=False,
        thick_quadrupole=True, thick_steerers=False, implementation="vectorized",
    )
    assert legacy[0].shape == legacy[1].shape == (len(bpms), len(correctors), len(subset))
    np.testing.assert_allclose(vectorized[0], legacy[0], rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(vectorized[1], legacy[1], rtol=2e-14, atol=2e-14)


def test_analytical_implementation_is_validated_and_identified_in_timing():
    ring, bpms, correctors, quadrupoles = _ring_and_indices()
    timing = []
    normal_module.analytic_orm_variation_with_normal_quadrupole(
        ring, bpms, correctors, quadrupoles[:1], verbose=False,
        implementation="legacy", timing_callback=timing.append,
    )
    assert timing and all(item["analytical_implementation"] == "legacy" for item in timing)
    import pytest
    with pytest.raises(ValueError, match="legacy.*vectorized"):
        normal_module.analytic_orm_variation_with_normal_quadrupole(
            ring, bpms, correctors, quadrupoles, implementation="fastest"
        )


def test_dispersion_enabled_jacobian_propagates_both_implementations(monkeypatch, tmp_path):
    ring, bpms, correctors, quadrupoles = _ring_and_indices()
    seen = []

    def fake_formula(_ring, ind_bpms, ind_cors, ind_quads, *, implementation, **_kwargs):
        seen.append(implementation)
        shape = (len(ind_bpms), len(ind_cors), len(ind_quads))
        values = np.arange(np.prod(shape), dtype=float).reshape(shape)
        return values, values + 1.0

    def fake_dispersion(**kwargs):
        return np.full((len(kwargs["quads_ind"]), 2 * len(kwargs["bpm_indexes"])), 7.0), None

    monkeypatch.setattr(pyloco_module, "analytic_orm_variation_with_normal_quadrupole", fake_formula)
    monkeypatch.setattr(pyloco_module, "calculate_quads_dispersion_jacobian", fake_dispersion)
    results = []
    saved_implementations = []
    for implementation in ("legacy", "vectorized"):
        jacobian, *_ = pyloco_module.compute_jacobian(
            ring, C_model=np.zeros((2 * len(bpms), 2 * len(correctors) + 1)),
            dkick=(np.ones(len(correctors)), np.ones(len(correctors))),
            bpm_indexes=bpms, CMords=(correctors, correctors),
            quads_ind=quadrupoles, nHorCOR=len(correctors), nVerCOR=len(correctors),
            nHBPM=len(bpms), nVBPM=len(bpms), C=np.eye(2 * len(bpms)),
            dk=1e-6, CAVords=[],
            includeDispersion=True, include_quads=True,
            quad_jacobian_calculator="Analytical",
            analytical_implementation=implementation, output_dir=tmp_path,
            save_jacobians=True,
        )
        results.append(jacobian)
        import h5py
        saved = list((tmp_path / "jacobians" / "quads").glob("*.h5"))
        assert len(saved) == 1
        with h5py.File(saved[0], "r") as handle:
            saved_implementations.append(handle.attrs["analytical_implementation"])
            assert int(handle.attrs["analytical_worker_count"]) == 1
    assert seen == ["legacy", "legacy", "vectorized", "vectorized"]
    np.testing.assert_array_equal(results[0], results[1])
    assert results[0].shape == (
        len(quadrupoles), 2 * len(bpms), 2 * len(correctors) + 1
    )
    assert saved_implementations == ["legacy", "vectorized"]


def test_skew_multiprocessing_accepts_python_integer_ordinals(monkeypatch):
    ring, bpms, correctors, quadrupoles = _ring_and_indices()
    monkeypatch.setattr(skew_module.multiprocessing, "Pool", _SerialPool)
    expected = skew_module.analytic_orm_variation_with_skew_quadrupole(
        ring, bpms, correctors, quadrupoles, verbose=False,
        thick_skew=True, thick_steerer=False, use_mp=False)
    actual = skew_module.analytic_orm_variation_with_skew_quadrupole(
        ring, bpms, correctors, quadrupoles, verbose=False,
        thick_skew=True, thick_steerer=False, use_mp=True)
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])


def test_analytical_multiprocessing_cancel_terminates_pool(monkeypatch):
    ring, bpms, correctors, quadrupoles = _ring_and_indices()
    state = {"terminated": False}

    class Pending:
        def get(self, timeout=None):
            raise normal_module.multiprocessing.TimeoutError

    class CancelPool(_SerialPool):
        def starmap_async(self, function, arguments):
            return Pending()

        def terminate(self):
            state["terminated"] = True

    monkeypatch.setattr(normal_module.multiprocessing, "Pool", CancelPool)
    import pytest
    with pytest.raises(RuntimeError, match="cancelled during analytical normal"):
        normal_module.analytic_orm_variation_with_normal_quadrupole(
            ring, bpms, correctors, quadrupoles, verbose=False,
            thick_quadrupole=True, thick_steerers=False, use_mp=True,
            cancel_callback=lambda: True)
    assert state["terminated"]
