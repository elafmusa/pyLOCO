import at
import numpy as np

from pyLOCO.config import FitInitConfig, RMConfig
from pyLOCO.pyloco import (
    calculate_skew_jacobian_analytical,
    compute_jacobian,
    pyloco,
)
from pyLOCO.response_matrix import response_matrix


def _ring_and_indices():
    half_drift = at.Drift("Dr", 0.25)
    drift = at.Drift("Dr", 0.5)
    bend = at.Dipole("Bend", 1.0, 2 * np.pi / 40)
    cell = at.Lattice(
        [
            half_drift, bend, drift, at.Monitor("BPM_F"),
            at.Corrector("HCOR_F", 0.0, [0.0, 0.0]),
            at.Corrector("VCOR_F", 0.0, [0.0, 0.0]),
            at.Quadrupole("QF", 0.5, 1.2), drift, bend, drift,
            at.Monitor("BPM_D"),
            at.Corrector("HCOR_D", 0.0, [0.0, 0.0]),
            at.Corrector("VCOR_D", 0.0, [0.0, 0.0]),
            at.Quadrupole("QD", 0.5, -1.2), half_drift,
        ],
        energy=1e9,
    )
    ring = cell * 20
    bpm = np.asarray(at.get_refpts(ring, at.Monitor), dtype=int)
    hcor = np.asarray(at.get_refpts(ring, "HCOR*"), dtype=int)
    vcor = np.asarray(at.get_refpts(ring, "VCOR*"), dtype=int)
    skews = np.asarray(at.get_refpts(ring, at.Quadrupole), dtype=int)
    return ring, bpm, hcor, vcor, skews


def _orm(ring, bpm, hcor, vcor, kick=1e-5, include_dispersion=False):
    return response_matrix(
        ring,
        config=RMConfig(
            bpm_ords=bpm,
            cm_ords=[hcor, vcor],
            dkick=[[kick] * len(hcor), [kick] * len(vcor)],
            calculator="Linear",
            bidirectional=True,
            includeDispersion=include_dispersion,
            fixedpathlength=False,
        ),
    )


def _central_skew_derivative(ring, index, bpm, hcor, vcor, step=1e-6):
    return _central_skew_group_derivative(ring, [index], bpm, hcor, vcor, step)


def _central_skew_group_derivative(ring, indices, bpm, hcor, vcor, step=1e-6):
    nominal = [float(ring[index].PolynomA[1]) for index in indices]
    try:
        for index, value in zip(indices, nominal):
            ring[index].PolynomA[1] = value + step
        plus = _orm(ring, bpm, hcor, vcor)
        for index, value in zip(indices, nominal):
            ring[index].PolynomA[1] = value - step
        minus = _orm(ring, bpm, hcor, vcor)
    finally:
        for index, value in zip(indices, nominal):
            ring[index].PolynomA[1] = value
    return (plus - minus) / (2.0 * step)


def test_individual_skew_full_four_block_jacobian():
    ring, bpm, hcor, vcor, skews = _ring_and_indices()
    skew = int(skews[3])
    kick = 1e-5
    analytical, delta = calculate_skew_jacobian_analytical(
        ring=ring,
        C_model=_orm(ring, bpm, hcor, vcor, kick=kick),
        dkick=[[kick] * len(hcor), [kick] * len(vcor)],
        used_cor_ind=[hcor, vcor],
        bpm_indexes=bpm,
        skew_ind=[skew],
        C=np.eye(2 * len(bpm)),
        fit_cfg=FitInitConfig(),
        analytical_thick_skew=True,
        analytical_thick_steerers=False,
        analytical_verbose=False,
        analytical_use_mp=False,
    )
    numerical = _central_skew_derivative(ring, skew, bpm, hcor, vcor)
    assert delta is None
    assert analytical.shape == (1, 2 * len(bpm), len(hcor) + len(vcor))

    blocks = {
        "XX": (slice(None, len(bpm)), slice(None, len(hcor))),
        "XY": (slice(None, len(bpm)), slice(len(hcor), None)),
        "YX": (slice(len(bpm), None), slice(None, len(hcor))),
        "YY": (slice(len(bpm), None), slice(len(hcor), None)),
    }
    for name, (rows, columns) in blocks.items():
        actual = analytical[0, rows, columns]
        expected = numerical[rows, columns]
        if name in {"XX", "YY"}:
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-13)
        else:
            relative_error = np.linalg.norm(expected - actual) / np.linalg.norm(expected)
            correlation = np.corrcoef(actual.ravel(), expected.ravel())[0, 1]
            # Preserve visibility of the accepted ~0.17% thick-skew
            # analytical residual; no empirical rescaling is applied.
            assert relative_error < 0.002
            assert correlation > 0.99999


def test_skew_family_matches_numerical_and_sum_of_individuals():
    ring, bpm, hcor, vcor, skews = _ring_and_indices()
    members = [int(skews[3]), int(skews[11])]
    kick = 1e-5
    common = dict(
        ring=ring,
        C_model=_orm(ring, bpm, hcor, vcor, kick=kick),
        dkick=[[kick] * len(hcor), [kick] * len(vcor)],
        used_cor_ind=[hcor, vcor],
        bpm_indexes=bpm,
        C=np.eye(2 * len(bpm)),
        fit_cfg=FitInitConfig(),
        analytical_thick_skew=True,
        analytical_thick_steerers=False,
        analytical_verbose=False,
        analytical_use_mp=False,
    )
    individuals, _ = calculate_skew_jacobian_analytical(
        skew_ind=members, **common
    )
    family, _ = calculate_skew_jacobian_analytical(
        skew_ind=[members], **common
    )
    np.testing.assert_array_equal(family[0], individuals.sum(axis=0))

    numerical = _central_skew_group_derivative(
        ring, members, bpm, hcor, vcor
    )
    n_bpm = len(bpm)
    n_hcor = len(hcor)
    for name, rows, columns in (
        ("XX", slice(None, n_bpm), slice(None, n_hcor)),
        ("XY", slice(None, n_bpm), slice(n_hcor, None)),
        ("YX", slice(n_bpm, None), slice(None, n_hcor)),
        ("YY", slice(n_bpm, None), slice(n_hcor, None)),
    ):
        actual = family[0, rows, columns]
        expected = numerical[rows, columns]
        if name in {"XX", "YY"}:
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-13)
        else:
            assert np.linalg.norm(expected - actual) / np.linalg.norm(expected) < 0.002
            assert np.corrcoef(actual.ravel(), expected.ravel())[0, 1] > 0.99999


def _computed_skew_jacobian(
    ring, bpm, hcor, vcor, skew_parameters, *, individuals, method, output_dir
):
    kick = 1e-5
    model = _orm(
        ring, bpm, hcor, vcor, kick=kick, include_dispersion=True
    )
    jacobian, _, delta_skew, _ = compute_jacobian(
        ring=ring,
        C_model=model,
        dkick=[[kick] * len(hcor), [kick] * len(vcor)],
        dk=None,
        bpm_indexes=bpm,
        CMords=[hcor, vcor],
        quads_ind=[],
        nHorCOR=len(hcor),
        nVerCOR=len(vcor),
        nHBPM=len(bpm),
        nVBPM=len(bpm),
        C=np.eye(2 * len(bpm)),
        CAVords=[],
        skew_ind=skew_parameters,
        includeDispersion=True,
        delta_skew_=1e-5,
        include_quads=False,
        include_skew=True,
        skew_individuals=individuals,
        auto_correct_delta=False,
        HCMCoupling=np.zeros(len(hcor)),
        VCMCoupling=np.zeros(len(vcor)),
        fit_cfg=FitInitConfig(),
        skew_jacobian_calculator=method,
        force_recompute=True,
        output_dir=output_dir,
    )
    return jacobian, delta_skew


def test_hybrid_skew_dispersion_individual_and_family(tmp_path):
    ring, bpm, hcor, vcor, skews = _ring_and_indices()
    individual = [int(skews[3])]
    family = [[int(skews[3]), int(skews[11])]]
    for label, parameters, individuals in (
        ("individual", individual, True),
        ("family", family, False),
    ):
        numerical, numerical_delta = _computed_skew_jacobian(
            ring, bpm, hcor, vcor, parameters, individuals=individuals,
            method="Numerical", output_dir=tmp_path / label / "numerical",
        )
        hybrid, hybrid_delta = _computed_skew_jacobian(
            ring, bpm, hcor, vcor, parameters, individuals=individuals,
            method="Analytical", output_dir=tmp_path / label / "analytical",
        )
        orm_error = np.linalg.norm(numerical[:, :, :-1] - hybrid[:, :, :-1])
        orm_error /= np.linalg.norm(numerical[:, :, :-1])
        assert orm_error < 0.002
        np.testing.assert_array_equal(hybrid[:, :, -1], numerical[:, :, -1])
        np.testing.assert_array_equal(hybrid_delta, numerical_delta)


def test_two_iteration_skew_loco_numerical_vs_analytical(tmp_path):
    nominal, bpm, hcor, vcor, skews = _ring_and_indices()
    skew = int(skews[3])
    injected_strength = 5e-4
    measured_ring = nominal.deepcopy()
    measured_ring[skew].PolynomA[1] = injected_strength
    measured = _orm(
        measured_ring, bpm, hcor, vcor, include_dispersion=True
    )

    results = {}
    for method in ("Numerical", "Analytical"):
        output = tmp_path / method.lower()
        fit_results, fit_dict, fitted_ring, final_orm, _, chi2, _, _ = pyloco(
            nominal.deepcopy(),
            algorithm="gn",
            nIter=2,
            used_bpms_ords=bpm,
            used_cor_ords=[hcor, vcor],
            quads_ords=[],
            skew_ords=[skew],
            CAVords=[],
            quads_tilt_ind=[],
            nHBPM=len(bpm),
            nVBPM=len(bpm),
            nHorCOR=len(hcor),
            nVerCOR=len(vcor),
            orm_measured=measured,
            weights=np.ones((2 * len(bpm), 1)),
            includeDispersion=True,
            measured_eta_x=measured[:len(bpm), -1],
            measured_eta_y=measured[len(bpm):, -1],
            CMstep=[[1e-5] * len(hcor), [1e-5] * len(vcor)],
            rfStep=-3000.0,
            fit_list=("skew_quads",),
            skew_individuals=True,
            remove_coupling_=False,
            auto_correct_delta=False,
            fixedpathlength=False,
            fit_cfg=FitInitConfig(fit_list=("skew_quads",)),
            skew_jacobian_calculator=method,
            svd_selection_method="threshold",
            svd_threshold=1e-12,
            show_svd_plot=False,
            force_recompute=True,
            output_dir=output,
        )
        recovered = float(fit_dict[1]["skew_quads"][0])
        assert abs(recovered - injected_strength) < 2e-6
        assert len(chi2) == 2
        assert chi2[1] <= chi2[0]
        for iteration in (1, 2):
            matches = list(
                (output / "jacobians" / "skew").glob(
                    f"J_skew_{method.lower()}_iter{iteration}_*.h5"
                )
            )
            assert len(matches) == 1
        results[method] = dict(
            recovered=recovered,
            chi2=np.asarray(chi2),
            final_orm=final_orm,
            final_dispersion=final_orm[:, -1],
            ring_strength=float(fitted_ring[skew].PolynomA[1]),
        )

    np.testing.assert_allclose(
        results["Analytical"]["recovered"],
        results["Numerical"]["recovered"],
        rtol=5e-3,
    )
    np.testing.assert_allclose(
        results["Analytical"]["final_orm"],
        results["Numerical"]["final_orm"],
        rtol=2e-3,
        atol=1e-12,
    )
    orm_difference = (
        results["Analytical"]["final_orm"][:, :-1]
        - results["Numerical"]["final_orm"][:, :-1]
    )
    dispersion_difference = (
        results["Analytical"]["final_dispersion"]
        - results["Numerical"]["final_dispersion"]
    )
    print(
        "two-iteration summary:",
        {
            method: {
                "chi2": results[method]["chi2"].tolist(),
                "recovered": results[method]["recovered"],
            }
            for method in ("Numerical", "Analytical")
        },
        "final_orm_rms_difference=", float(np.sqrt(np.mean(orm_difference**2))),
        "final_orm_max_difference=", float(np.max(np.abs(orm_difference))),
        "final_dispersion_rms_difference=", float(np.sqrt(np.mean(dispersion_difference**2))),
        "final_dispersion_max_difference=", float(np.max(np.abs(dispersion_difference))),
    )
