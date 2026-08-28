import numpy as np
import at

import pyLOCO.analytic_orm_with_normal_quad_errors as normal_module
import pyLOCO.analytic_orm_with_skew_quad_errors as skew_module


class _SerialPool:
    """Exercise Pool argument packing deterministically in the test process."""

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
