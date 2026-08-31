import at
import numpy as np

import pyLOCO.pyloco as pyloco_module
from pyLOCO.config import FitInitConfig, RMConfig
from pyLOCO.pyloco import calculate_quads_dispersion_jacobian
from pyLOCO.response_matrix import calculate_rf_response, response_matrix


def _fodo_ring_with_correctors():
    half_drift = at.Drift("Dr", 0.25)
    drift = at.Drift("Dr", 0.5)
    bend = at.Dipole("Bend", 1.0, 2 * np.pi / 40)
    cell = at.Lattice(
        [
            half_drift,
            bend,
            drift,
            at.Monitor("BPM_F"),
            at.Corrector("HCOR_F", 0.0, [0.0, 0.0]),
            at.Corrector("VCOR_F", 0.0, [0.0, 0.0]),
            at.Quadrupole("QF", 0.5, 1.2),
            drift,
            bend,
            drift,
            at.Monitor("BPM_D"),
            at.Corrector("HCOR_D", 0.0, [0.0, 0.0]),
            at.Corrector("VCOR_D", 0.0, [0.0, 0.0]),
            at.Quadrupole("QD", 0.5, -1.2),
            half_drift,
        ],
        energy=1e9,
    )
    return cell * 20


def test_tracking_orm_supports_at_corrector_and_restores_kicks():
    ring = _fodo_ring_with_correctors()
    bpm = np.asarray(at.get_refpts(ring, at.Monitor), dtype=int)
    hcor = np.asarray(at.get_refpts(ring, "HCOR*"), dtype=int)
    vcor = np.asarray(at.get_refpts(ring, "VCOR*"), dtype=int)
    kick = 10e-6

    def config(calculator):
        return RMConfig(
            bpm_ords=bpm,
            cm_ords=[hcor, vcor],
            dkick=[[kick] * len(hcor), [kick] * len(vcor)],
            calculator=calculator,
            bidirectional=True,
            includeDispersion=False,
            fixedpathlength=False,
        )

    orm_linear = response_matrix(ring, config=config("Linear"))
    orm_tracking = response_matrix(ring, config=config("Tracking"))
    orm_numerical = response_matrix(ring, config=config("Numerical"))

    np.testing.assert_allclose(orm_tracking, orm_linear, rtol=1e-9, atol=1e-15)
    np.testing.assert_array_equal(orm_tracking, orm_numerical)
    for index in np.concatenate((hcor, vcor)):
        np.testing.assert_array_equal(ring[index].KickAngle, [0.0, 0.0])


def test_analytical_calculator_reaches_uncoupled_implementation():
    ring = _fodo_ring_with_correctors()
    bpm = np.asarray(at.get_refpts(ring, at.Monitor), dtype=int)
    hcor = np.asarray(at.get_refpts(ring, "HCOR*"), dtype=int)
    vcor = np.asarray(at.get_refpts(ring, "VCOR*"), dtype=int)
    kick = 10e-6
    common = dict(
        bpm_ords=bpm, cm_ords=[hcor, vcor],
        dkick=[[kick] * len(hcor), [kick] * len(vcor)],
        bidirectional=True, includeDispersion=False, fixedpathlength=False,
    )
    analytical = response_matrix(ring, config=RMConfig(calculator="Analytical", **common))
    linear = response_matrix(ring, config=RMConfig(calculator="Linear", **common))

    np.testing.assert_allclose(analytical, linear, rtol=1e-9, atol=1e-15)
    with np.testing.assert_raises_regex(ValueError, "uncoupled"):
        response_matrix(ring, config=RMConfig(calculator="Analytical", coupling_orm=True, **common))


def test_dispersion_only_response_matches_full_linear_response_column():
    ring = _fodo_ring_with_correctors()
    bpm = np.asarray(at.get_refpts(ring, at.Monitor), dtype=int)
    hcor = np.asarray(at.get_refpts(ring, "HCOR*"), dtype=int)
    vcor = np.asarray(at.get_refpts(ring, "VCOR*"), dtype=int)
    kick = 10e-6
    config = RMConfig(
        bpm_ords=bpm,
        cm_ords=[hcor, vcor],
        cav_ords=np.asarray([], dtype=int),
        dkick=[[kick] * len(hcor), [kick] * len(vcor)],
        calculator="Linear",
        bidirectional=True,
        includeDispersion=True,
        rfStep=-3000.0,
        fixedpathlength=False,
    )
    full = response_matrix(ring, config=config)
    dispersion = calculate_rf_response(
        ring, bpm, config.cav_ords, config.rfStep,
        calculator=config.calculator,
        bidirectional=config.bidirectional,
        frequency=config.Frequency,
        harm_number=config.HarmNumber,
        rf_attr=config.RFAttr,
    )
    np.testing.assert_array_equal(dispersion, full[:, -1])


def test_tracking_rf_response_is_zero_in_current_4d_path():
    ring = _fodo_ring_with_correctors()
    ring.append(at.RFCavity("RFC", 0.0, 3e6, 5e8, 100, 1e9))
    ring.disable_6d()
    bpm = np.asarray(at.get_refpts(ring, at.Monitor), dtype=int)
    cavity = np.asarray(at.get_refpts(ring, at.elements.RFCavity), dtype=int)

    linear = calculate_rf_response(
        ring, bpm, cavity, 40.0, calculator="Linear", frequency=5e8,
    )
    tracking = calculate_rf_response(
        ring, bpm, cavity, 40.0, calculator="Tracking", frequency=5e8,
    )

    assert np.linalg.norm(linear) > 0.0
    np.testing.assert_array_equal(tracking, np.zeros_like(tracking))


def test_linear_and_analytical_normal_quad_dispersion_derivatives_match():
    ring = _fodo_ring_with_correctors()
    bpm = np.asarray(at.get_refpts(ring, at.Monitor), dtype=int)
    hcor = np.asarray(at.get_refpts(ring, "HCOR*"), dtype=int)[:2]
    vcor = np.asarray(at.get_refpts(ring, "VCOR*"), dtype=int)[:2]
    quads = np.asarray(at.get_refpts(ring, at.elements.Quadrupole), dtype=int)[:2]
    kicks = (np.full(len(hcor), 1e-5), np.full(len(vcor), 1e-5))
    common_config = dict(
        bpm_ords=bpm, cm_ords=(hcor, vcor), cav_ords=np.asarray([], dtype=int),
        dkick=kicks, includeDispersion=True, rfStep=40.0,
    )
    results = {}
    for calculator in ("Linear", "Analytical"):
        model = response_matrix(
            ring, config=RMConfig(calculator=calculator, **common_config)
        )
        results[calculator] = calculate_quads_dispersion_jacobian(
            ring=ring, C_model=model, dkick=kicks,
            used_cor_ind=(hcor, vcor), bpm_indexes=bpm,
            quads_ind=quads, dk=None, C=np.eye(2 * len(bpm)), individuals=True,
            HCMCoupling=np.zeros(len(hcor)), VCMCoupling=np.zeros(len(vcor)),
            rf_step=40.0, CAVords=np.asarray([], dtype=int),
            auto_correct_delta=True, fit_cfg=FitInitConfig(),
            orm_calculator=calculator, use_mp=False,
        )

    np.testing.assert_allclose(
        results["Linear"][1], results["Analytical"][1], rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        results["Linear"][0], results["Analytical"][0], rtol=2e-12, atol=1e-15
    )


def test_adaptive_step_selection_has_finite_iteration_limit(monkeypatch):
    ring = _fodo_ring_with_correctors()
    bpm = np.asarray(at.get_refpts(ring, at.Monitor), dtype=int)
    hcor = np.asarray(at.get_refpts(ring, "HCOR*"), dtype=int)[:2]
    vcor = np.asarray(at.get_refpts(ring, "VCOR*"), dtype=int)[:2]
    quad = int(np.asarray(at.get_refpts(ring, at.elements.Quadrupole), dtype=int)[0])
    rows = 2 * len(bpm)
    columns = len(hcor) + len(vcor) + 1
    calls = []

    def mismatched_response(_ring, *, config):
        calls.append(config.calculator)
        return np.ones((rows, columns))

    monkeypatch.setattr(pyloco_module, "response_matrix", mismatched_response)
    monkeypatch.setattr(pyloco_module, "G_C", np.eye(rows))
    monkeypatch.setattr(pyloco_module, "G_CMODEL", np.zeros((rows, columns)))

    import pytest
    with pytest.raises(RuntimeError, match=(
        r"did not converge after 25 iterations.*calculator=Linear.*"
        r"group=.*RMSDelta=.*target_range="
    )):
        pyloco_module.generating_quads_response_matrices(
            quad, ring, (np.full(len(hcor), 1e-5), np.full(len(vcor), 1e-5)),
            (hcor, vcor), bpm, None, True,
            np.zeros(len(hcor)), np.zeros(len(vcor)), 40.0, True,
            "quads", FitInitConfig(), True, "Linear", np.asarray([], dtype=int), 1,
        )

    assert calls == ["Linear"] * pyloco_module.MAX_ADAPTIVE_STEP_ITERATIONS
