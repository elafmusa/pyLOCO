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


def _replace_first_correctors(ring, horizontal, vertical):
    h_index = int(np.asarray(at.get_refpts(ring, "HCOR*"), dtype=int)[0])
    v_index = int(np.asarray(at.get_refpts(ring, "VCOR*"), dtype=int)[0])
    ring[h_index] = horizontal
    ring[v_index] = vertical
    return h_index, v_index


def _two_corrector_orm(ring, indices, calculator, kick=10e-6):
    bpm = np.asarray(at.get_refpts(ring, at.Monitor), dtype=int)
    return response_matrix(
        ring,
        config=RMConfig(
            bpm_ords=bpm,
            cm_ords=[[indices[0]], [indices[1]]],
            dkick=[[kick], [kick]],
            calculator=calculator,
            bidirectional=True,
            includeDispersion=False,
            fixedpathlength=False,
        ),
    )


def test_tracking_orm_converts_distinct_finite_multipole_lengths_to_integrated_kicks():
    ring = _fodo_ring_with_correctors()
    horizontal = at.Multipole("MH", 0.23, [0.0], [0.0])
    vertical = at.Multipole("MV", 0.41, [0.0], [0.0])
    indices = _replace_first_correctors(ring, horizontal, vertical)

    linear = _two_corrector_orm(ring, indices, "Linear")
    tracking = _two_corrector_orm(ring, indices, "Tracking")

    np.testing.assert_allclose(tracking, linear, rtol=5e-11, atol=5e-15)
    assert np.vdot(tracking[:, 0], linear[:, 0]) > 0.0
    assert np.vdot(tracking[:, 1], linear[:, 1]) > 0.0
    np.testing.assert_array_equal(horizontal.PolynomB, [0.0])
    np.testing.assert_array_equal(vertical.PolynomA, [0.0])


def test_tracking_orm_uses_integrated_polynomials_for_thin_correctors():
    ring = _fodo_ring_with_correctors()
    horizontal = at.ThinMultipole("TH", [0.0], [0.0])
    vertical = at.ThinMultipole("TV", [0.0], [0.0])
    indices = _replace_first_correctors(ring, horizontal, vertical)

    linear = _two_corrector_orm(ring, indices, "Linear")
    tracking = _two_corrector_orm(ring, indices, "Tracking")

    np.testing.assert_allclose(tracking, linear, rtol=5e-11, atol=5e-15)
    np.testing.assert_array_equal(horizontal.PolynomB, [0.0])
    np.testing.assert_array_equal(vertical.PolynomA, [0.0])


def test_tracking_orm_rejects_zero_length_thick_multipole_representation():
    ring = _fodo_ring_with_correctors()
    invalid = at.Multipole(
        "BAD", 0.0, [0.0], [0.0], PassMethod="StrMPoleSymplectic4Pass"
    )
    vertical = at.ThinMultipole("TV", [0.0], [0.0])
    indices = _replace_first_correctors(ring, invalid, vertical)

    with np.testing.assert_raises_regex(TypeError, "no integrated-kick representation"):
        _two_corrector_orm(ring, indices, "Tracking")


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


def test_tracking_rf_response_matches_linear_with_disabled_4d_cavity():
    ring = _fodo_ring_with_correctors()
    ring.append(at.RFCavity("RFC", 0.0, 3e6, 5e8, 100, 1e9))
    ring.disable_6d()
    bpm = np.asarray(at.get_refpts(ring, at.Monitor), dtype=int)
    cavity = np.asarray(at.get_refpts(ring, at.elements.RFCavity), dtype=int)

    linear = calculate_rf_response(
        ring, bpm, cavity, 40.0, calculator="Linear", frequency=5e8,
        harm_number=100,
    )
    tracking = calculate_rf_response(
        ring, bpm, cavity, 40.0, calculator="Tracking", frequency=5e8,
        harm_number=100,
    )
    hcor = np.asarray(at.get_refpts(ring, "HCOR*"), dtype=int)
    vcor = np.asarray(at.get_refpts(ring, "VCOR*"), dtype=int)
    full_tracking = response_matrix(ring, config=RMConfig(
        bpm_ords=bpm, cm_ords=(hcor, vcor), cav_ords=cavity,
        dkick=(np.full(len(hcor), 1e-5), np.full(len(vcor), 1e-5)),
        calculator="Tracking", bidirectional=True, includeDispersion=True,
        rfStep=40.0, Frequency=5e8, HarmNumber=100,
        fixedpathlength=False,
    ))

    assert np.linalg.norm(linear[:len(bpm)]) > 0.0
    assert np.all(np.isfinite(tracking))
    np.testing.assert_allclose(tracking, linear, rtol=2e-12, atol=1e-15)
    np.testing.assert_allclose(full_tracking[:, -1], linear, rtol=2e-12, atol=1e-15)
    assert ring.is_6d is False
    assert ring[int(cavity[0])].PassMethod == "IdentityPass"
    assert ring[int(cavity[0])].Frequency == 5e8


def test_tracking_rf_response_uses_central_frequency_shift_in_active_6d_ring(
        monkeypatch):
    ring = _fodo_ring_with_correctors()
    ring.append(at.RFCavity("RFC", 0.0, 3e6, 5e8, 100, 1e9))
    bpm = np.asarray(at.get_refpts(ring, at.Monitor), dtype=int)
    cavity = np.asarray(at.get_refpts(ring, at.elements.RFCavity), dtype=int)
    initial_state = (ring.is_6d, ring[int(cavity[0])].PassMethod,
                     ring[int(cavity[0])].Frequency)
    seen_frequencies = []

    def fake_find_orbit6(test_ring, *, refpts):
        frequency = test_ring[int(cavity[0])].Frequency
        seen_frequencies.append(frequency)
        orbit = np.zeros((len(refpts), 6))
        orbit[:, 0] = frequency - initial_state[2]
        orbit[:, 2] = 2.0 * (frequency - initial_state[2])
        return np.zeros(6), orbit

    monkeypatch.setattr(at, "find_orbit6", fake_find_orbit6)
    tracking = calculate_rf_response(
        ring, bpm, cavity, 40.0, calculator="Tracking", frequency=5e8,
        bidirectional=True,
    )

    assert seen_frequencies == [5e8 + 20.0, 5e8 - 20.0]
    np.testing.assert_array_equal(tracking[:len(bpm)], np.full(len(bpm), 40.0))
    np.testing.assert_array_equal(tracking[len(bpm):], np.full(len(bpm), 80.0))
    assert (ring.is_6d, ring[int(cavity[0])].PassMethod,
            ring[int(cavity[0])].Frequency) == initial_state


def test_tracking_rf_response_is_finite_nonzero_with_active_6d_cavity():
    ring = _fodo_ring_with_correctors()
    ring.append(at.RFCavity("RFC", 0.0, 3e6, 5e8, 100, 1e9))
    bpm = np.asarray(at.get_refpts(ring, at.Monitor), dtype=int)
    cavity = np.asarray(at.get_refpts(ring, at.elements.RFCavity), dtype=int)
    initial_state = (ring.is_6d, ring[int(cavity[0])].PassMethod,
                     ring[int(cavity[0])].Frequency)

    tracking = calculate_rf_response(
        ring, bpm, cavity, 40.0, calculator="Tracking", frequency=5e8,
        harm_number=100, bidirectional=True,
    )

    assert np.all(np.isfinite(tracking))
    assert np.linalg.norm(tracking[:len(bpm)]) > 0.0
    assert (ring.is_6d, ring[int(cavity[0])].PassMethod,
            ring[int(cavity[0])].Frequency) == initial_state


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


def test_rf_only_dispersion_worker_matches_legacy_full_orm_column(monkeypatch):
    ring = _fodo_ring_with_correctors()
    ring.append(at.RFCavity("RFC", 0.0, 3e6, 5e8, 100, 1e9))
    ring.disable_6d()
    bpm = np.asarray(at.get_refpts(ring, at.Monitor), dtype=int)
    hcor = np.asarray(at.get_refpts(ring, "HCOR*"), dtype=int)[:2]
    vcor = np.asarray(at.get_refpts(ring, "VCOR*"), dtype=int)[:2]
    cavity = np.asarray(at.get_refpts(ring, at.elements.RFCavity), dtype=int)
    skew = int(np.asarray(at.get_refpts(ring, at.elements.Quadrupole), dtype=int)[0])
    kicks = (np.full(len(hcor), 1e-5), np.full(len(vcor), 1e-5))
    calibration = np.eye(2 * len(bpm))
    fit = FitInitConfig(fit_list=("skew_quads",), individuals=True)
    model = response_matrix(
        ring, config=RMConfig(
            bpm_ords=bpm, cm_ords=(hcor, vcor), cav_ords=cavity,
            dkick=kicks, includeDispersion=True, rfStep=40.0,
            calculator="Linear", Frequency=5e8,
        )
    )
    monkeypatch.setattr(pyloco_module, "G_C", calibration)
    monkeypatch.setattr(pyloco_module, "G_CMODEL", model)

    legacy, legacy_step, _ = pyloco_module.generating_quads_response_matrices(
        skew, ring, kicks, (hcor, vcor), bpm, 1e-3, True,
        np.zeros(len(hcor)), np.zeros(len(vcor)), 40.0, True,
        "skew_quads", fit, True, "Linear", cavity, 1,
    )
    rf_only, rf_step = pyloco_module._dispersion_rf_only_worker(
        skew, 1e-3, ring, kicks, (hcor, vcor), bpm, True,
        np.zeros(len(hcor)), np.zeros(len(vcor)), 40.0, cavity, True,
        fit, "Linear", "skew_quads",
    )

    np.testing.assert_allclose(rf_only, legacy[:, -1], rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(rf_step, legacy_step[0], rtol=0.0, atol=0.0)


def test_forward_rf_only_reuses_nominal_column_and_preserves_step(monkeypatch):
    ring = _fodo_ring_with_correctors()
    ring.append(at.RFCavity("RFC", 0.0, 3e6, 5e8, 100, 1e9))
    ring.disable_6d()
    bpm = np.asarray(at.get_refpts(ring, at.Monitor), dtype=int)
    hcor = np.asarray(at.get_refpts(ring, "HCOR*"), dtype=int)[:2]
    vcor = np.asarray(at.get_refpts(ring, "VCOR*"), dtype=int)[:2]
    cavity = np.asarray(at.get_refpts(ring, at.elements.RFCavity), dtype=int)
    quad = int(np.asarray(at.get_refpts(ring, at.elements.Quadrupole), dtype=int)[0])
    kicks = (np.full(len(hcor), 1e-5), np.full(len(vcor), 1e-5))
    calibration = np.eye(2 * len(bpm))
    fit = FitInitConfig(fit_list=("quads",), individuals=True)
    model = response_matrix(
        ring, config=RMConfig(
            bpm_ords=bpm, cm_ords=(hcor, vcor), cav_ords=cavity,
            dkick=kicks, includeDispersion=True, rfStep=40.0,
            calculator="Linear", Frequency=5e8,
        )
    )
    monkeypatch.setattr(pyloco_module, "G_C", calibration)
    monkeypatch.setattr(pyloco_module, "G_CMODEL", model)
    original = pyloco_module.calculate_rf_response
    calls = []

    def counted_rf(*args, **kwargs):
        calls.append(kwargs.get("calculator"))
        return original(*args, **kwargs)

    monkeypatch.setattr(pyloco_module, "calculate_rf_response", counted_rf)
    central, central_step = pyloco_module._dispersion_rf_only_worker(
        quad, 1e-3, ring, kicks, (hcor, vcor), bpm, True,
        np.zeros(len(hcor)), np.zeros(len(vcor)), 40.0, cavity, False,
        fit, "Linear", "quads", "central",
    )
    central_calls = len(calls)
    calls.clear()
    reused, reused_step = pyloco_module._dispersion_rf_only_worker(
        quad, 1e-3, ring, kicks, (hcor, vcor), bpm, True,
        np.zeros(len(hcor)), np.zeros(len(vcor)), 40.0, cavity, False,
        fit, "Linear", "quads", "central",
        reuse_adaptive_plus_rf=True,
    )
    assert reused_step == central_step
    assert len(calls) == 1
    np.testing.assert_allclose(reused, central, rtol=0.0, atol=0.0)
    calls.clear()
    forward, forward_step = pyloco_module._dispersion_rf_only_worker(
        quad, 1e-3, ring, kicks, (hcor, vcor), bpm, True,
        np.zeros(len(hcor)), np.zeros(len(vcor)), 40.0, cavity, False,
        fit, "Linear", "quads", "forward",
    )
    assert central_step == forward_step
    assert central_calls == 2
    assert len(calls) == 1
    assert np.all(np.isfinite(central))
    assert np.all(np.isfinite(forward))


def test_numerical_normal_and_skew_mp_workers_preserve_explicit_rf_frequency(tmp_path):
    ring = _fodo_ring_with_correctors()
    ring.append(at.RFCavity("RFC", 0.0, 3e6, 5e8, 100, 1e9))
    ring.disable_6d()
    bpm = np.asarray(at.get_refpts(ring, at.Monitor), dtype=int)
    hcor = np.asarray(at.get_refpts(ring, "HCOR*"), dtype=int)[:2]
    vcor = np.asarray(at.get_refpts(ring, "VCOR*"), dtype=int)[:2]
    cavity = np.asarray(at.get_refpts(ring, at.elements.RFCavity), dtype=int)
    magnets = np.asarray(at.get_refpts(ring, at.elements.Quadrupole), dtype=int)[:2]
    kicks = (np.full(len(hcor), 1e-5), np.full(len(vcor), 1e-5))
    calibration = np.eye(2 * len(bpm))
    frequency = 5e8
    model = response_matrix(
        ring, config=RMConfig(
            bpm_ords=bpm, cm_ords=(hcor, vcor), cav_ords=cavity,
            dkick=kicks, includeDispersion=True, rfStep=40.0,
            calculator="Linear", Frequency=frequency,
        )
    )

    for block in ("quads", "skew_quads"):
        fit = FitInitConfig(fit_list=(block,), individuals=True)
        numerical, numerical_steps = pyloco_module.calculate_quads_jacobian(
            ring=ring, C_model=model, dkick=kicks,
            used_cor_ind=(hcor, vcor), bpm_indexes=bpm,
            quads_ind=magnets, dk=1e-3, C=calibration,
            individuals=True, HCMCoupling=np.zeros(len(hcor)),
            VCMCoupling=np.zeros(len(vcor)), rf_step=40.0,
            block=block, CAVords=cavity, auto_correct_delta=False,
            fit_cfg=fit, output_dir=tmp_path / block,
            includeDispersion=True, orm_calculator="Linear",
            Frequency=frequency, processes=2,
        )
        rf_only, rf_only_steps = calculate_quads_dispersion_jacobian(
            ring=ring, C_model=model, dkick=kicks,
            used_cor_ind=(hcor, vcor), bpm_indexes=bpm,
            quads_ind=magnets, dk=1e-3, C=calibration,
            individuals=True, HCMCoupling=np.zeros(len(hcor)),
            VCMCoupling=np.zeros(len(vcor)), rf_step=40.0,
            CAVords=cavity, auto_correct_delta=False, fit_cfg=fit,
            orm_calculator="Linear", use_mp=True, workers=2,
            block=block, mp_worker_mode="rf_only",
            difference="central", step_metric="full_orm",
            frequency=frequency, report=False,
        )
        np.testing.assert_array_equal(numerical_steps, rf_only_steps)
        np.testing.assert_array_equal(numerical[:, :, -1], rf_only)
