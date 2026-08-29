from types import SimpleNamespace

import numpy as np

import pyLOCO.pyloco as pyloco_module
from pyLOCO.config import FitInitConfig


def test_tracking_numerical_quad_perturbations_keep_cavity_ordinals(monkeypatch):
    cavity_ordinals = np.asarray([17], dtype=int)
    ring = [SimpleNamespace(PolynomB=np.asarray([0.0, 1.0]))]
    seen_cavities = []

    def fake_set_correction(ring, values, indices, **_kwargs):
        ring[0].PolynomB[1] = float(np.asarray(values).ravel()[0])

    def fake_response_matrix(ring, *, config):
        seen_cavities.append(np.asarray(config.cav_ords, dtype=int).copy())
        strength = float(ring[0].PolynomB[1])
        return np.asarray([[strength, 2.0 * strength], [strength, 2.0 * strength]])

    monkeypatch.setattr(pyloco_module, "set_correction", fake_set_correction)
    monkeypatch.setattr(pyloco_module, "response_matrix", fake_response_matrix)
    monkeypatch.setattr(pyloco_module, "G_C", np.eye(2))
    monkeypatch.setattr(pyloco_module, "G_CMODEL", np.asarray([[1.0, 2.0], [1.0, 2.0]]))

    jacobian, delta, _logs = pyloco_module.generating_quads_response_matrices(
        0,
        ring,
        [[1.0], []],
        [[0], []],
        [0],
        1.0e-3,
        True,
        [0.0],
        [],
        40.0,
        False,
        "quads",
        FitInitConfig().__dict__.copy(),
        True,
        "Tracking",
        cavity_ordinals,
    )

    assert jacobian.shape == (2, 2)
    np.testing.assert_allclose(delta, [1.0e-3])
    assert len(seen_cavities) == 3  # step selection, +dK, and -dK
    for received in seen_cavities:
        np.testing.assert_array_equal(received, cavity_ordinals)
