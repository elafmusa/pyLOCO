import at
import numpy as np

from pyLOCO.config import RMConfig
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
