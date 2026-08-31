import at
import h5py
import numpy as np

from pyLOCO.config import FitInitConfig, RMConfig
from pyLOCO.pyloco import (
    calculate_skew_jacobian_analytical,
    compute_jacobian,
    pyloco,
)
from pyLOCO.response_matrix import response_matrix
from pyLOCO.analytic_orm_with_skew_quad_errors import (
    analytic_orm_variation_with_skew_quadrupole,
)


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
    ring, bpm, hcor, vcor, skew_parameters, *, individuals, method, output_dir,
    implementation="vectorized",
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
        skew_analytical_implementation=implementation,
        force_recompute=True,
        output_dir=output_dir,
    )
    return jacobian, delta_skew


def test_legacy_and_vectorized_skew_formulas_match_thick_thin_and_subsets():
    ring, bpm, hcor, _, skews = _ring_and_indices()
    subset = skews[[11, 3, 7]]
    for thick_skew in (False, True):
        legacy = analytic_orm_variation_with_skew_quadrupole(
            ring, bpm[:8], hcor[:5], subset, verbose=False,
            thick_skew=thick_skew, thick_steerer=False,
            implementation="legacy",
        )
        vectorized = analytic_orm_variation_with_skew_quadrupole(
            ring, bpm[:8], hcor[:5], subset, verbose=False,
            thick_skew=thick_skew, thick_steerer=False,
            implementation="vectorized",
        )
        for legacy_block, vectorized_block in zip(legacy, vectorized):
            assert legacy_block.shape == (8, 5, 3)
            np.testing.assert_allclose(
                vectorized_block, legacy_block, rtol=5e-15, atol=3e-14
            )


def test_legacy_and_vectorized_full_skew_jacobian_match_with_dispersion(tmp_path):
    ring, bpm, hcor, vcor, skews = _ring_and_indices()
    parameters = [int(skews[3]), int(skews[11])]
    results = {}
    for implementation in ("legacy", "vectorized"):
        results[implementation], _ = _computed_skew_jacobian(
            ring, bpm, hcor, vcor, parameters, individuals=True,
            method="Analytical", implementation=implementation,
            output_dir=tmp_path / implementation,
        )
    legacy = results["legacy"]
    vectorized = results["vectorized"]
    assert legacy.shape == vectorized.shape == (
        len(parameters), 2 * len(bpm), len(hcor) + len(vcor) + 1
    )
    assert np.isfinite(legacy).sum() == np.isfinite(vectorized).sum() == legacy.size
    np.testing.assert_allclose(vectorized, legacy, rtol=5e-15, atol=3e-14)
    difference = vectorized - legacy
    assert np.sqrt(np.mean(difference**2)) < 1e-15
    assert np.linalg.norm(difference) / np.linalg.norm(legacy) < 1e-14
    for parameter in (0, len(parameters) - 1):
        np.testing.assert_allclose(
            vectorized[parameter], legacy[parameter], rtol=5e-15, atol=3e-14
        )


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
                save_jacobians=True,
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


def test_two_iteration_normal_and_skew_calculator_comparison(tmp_path):
    nominal, bpm, hcor, vcor, magnets = _ring_and_indices()
    quad_indices = [int(magnets[i]) for i in (2, 7, 12)]
    skew_indices = [int(magnets[i]) for i in (3, 8, 13)]
    injected_quad = np.asarray([8e-4, -6e-4, 4e-4])
    injected_skew = np.asarray([5e-4, -4e-4, 3e-4])
    nominal_quad = np.asarray(
        [float(nominal[index].PolynomB[1]) for index in quad_indices]
    )
    measured_ring = nominal.deepcopy()
    for index, delta in zip(quad_indices, injected_quad):
        measured_ring[index].PolynomB[1] += delta
    for index, delta in zip(skew_indices, injected_skew):
        measured_ring[index].PolynomA[1] += delta
    measured = _orm(
        measured_ring, bpm, hcor, vcor, include_dispersion=True
    )

    results = {}
    for method in ("Numerical", "Analytical"):
        output = tmp_path / f"normal-skew-{method.lower()}"
        initial_chi2 = []
        _, fit_dict, _, final_orm, _, chi2, _, _ = pyloco(
            nominal.deepcopy(),
            algorithm="gn",
            nIter=2,
            used_bpms_ords=bpm,
            used_cor_ords=[hcor, vcor],
            quads_ords=quad_indices,
            skew_ords=skew_indices,
            CAVords=[],
            quads_tilt_ind=[],
            nHBPM=len(bpm), nVBPM=len(bpm),
            nHorCOR=len(hcor), nVerCOR=len(vcor),
            orm_measured=measured,
            weights=np.ones((2 * len(bpm), 1)),
            includeDispersion=True,
            measured_eta_x=measured[:len(bpm), -1],
            measured_eta_y=measured[len(bpm):, -1],
            CMstep=[[1e-5] * len(hcor), [1e-5] * len(vcor)],
            rfStep=-3000.0,
            fit_list=("quads", "skew_quads"),
            quad_individuals=True,
            skew_individuals=True,
            remove_coupling_=False,
            auto_correct_delta=False,
            fixedpathlength=False,
            fit_cfg=FitInitConfig(
                fit_list=("quads", "skew_quads"), individuals=True
            ),
            quad_jacobian_calculator=method,
            skew_jacobian_calculator=method,
            svd_selection_method="threshold",
            svd_threshold=1e-12,
            show_svd_plot=False,
            force_recompute=True,
            initial_chi2_callback=initial_chi2.append,
                output_dir=output,
                save_jacobians=True,
        )
        recovered_quad = np.asarray(fit_dict[1]["quads"]) - nominal_quad
        recovered_skew = np.asarray(fit_dict[1]["skew_quads"])
        metadata = []
        for block, dataset, attr in (
            ("quads", "J_quads", "jacobian_calculator"),
            ("skew", "J_skew", "calculator"),
        ):
            for iteration in (1, 2):
                files = list(
                    (output / "jacobians" / block).glob(
                        f"J_{block}_{method.lower()}_iter{iteration}_*.h5"
                    )
                )
                assert len(files) == 1
                with h5py.File(files[0], "r") as handle:
                    assert dataset in handle
                    assert str(handle.attrs[attr]).lower() == method.lower()
                    if block == "skew":
                        expected_implementation = (
                            "vectorized" if method == "Analytical" else "not_applicable"
                        )
                        assert (
                            str(handle.attrs["skew_analytical_implementation"]).lower()
                            == expected_implementation.lower()
                        )
                    metadata.append(float(handle.attrs["computation_seconds"]))
        results[method] = {
            "initial_chi2": initial_chi2[0],
            "chi2": np.asarray(chi2),
            "quad": recovered_quad,
            "skew": recovered_skew,
            "orm": final_orm[:, :-1],
            "dispersion": final_orm[:, -1],
            "jacobian_seconds": np.asarray(metadata),
        }

    for method in ("Numerical", "Analytical"):
        np.testing.assert_allclose(
            results[method]["quad"], injected_quad, rtol=2e-3, atol=2e-7
        )
        np.testing.assert_allclose(
            results[method]["skew"], injected_skew, rtol=2e-3, atol=2e-7
        )
    np.testing.assert_allclose(
        results["Analytical"]["orm"], results["Numerical"]["orm"],
        rtol=3e-3, atol=2e-11,
    )
    np.testing.assert_allclose(
        results["Analytical"]["dispersion"],
        results["Numerical"]["dispersion"],
        rtol=3e-3, atol=2e-11,
    )
    def comparison_metrics(numerical, analytical, truth=None):
        difference = analytical - numerical
        scale = max(np.linalg.norm(numerical), np.finfo(float).eps)
        metrics = {
            "numerical_vs_analytical_relative": float(
                np.linalg.norm(difference) / scale
            ),
            "correlation": float(
                np.corrcoef(numerical.ravel(), analytical.ravel())[0, 1]
            ),
            "maximum_difference": float(np.max(np.abs(difference))),
        }
        if truth is not None:
            metrics["numerical_error_vs_truth"] = float(
                np.linalg.norm(numerical - truth)
            )
            metrics["analytical_error_vs_truth"] = float(
                np.linalg.norm(analytical - truth)
            )
        return metrics

    summary = {
        "initial_chi2": results["Numerical"]["initial_chi2"],
        "numerical_chi2": results["Numerical"]["chi2"].tolist(),
        "analytical_chi2": results["Analytical"]["chi2"].tolist(),
        "numerical_quad": results["Numerical"]["quad"].tolist(),
        "analytical_quad": results["Analytical"]["quad"].tolist(),
        "numerical_skew": results["Numerical"]["skew"].tolist(),
        "analytical_skew": results["Analytical"]["skew"].tolist(),
        "quad_metrics": comparison_metrics(
            results["Numerical"]["quad"], results["Analytical"]["quad"],
            injected_quad,
        ),
        "skew_metrics": comparison_metrics(
            results["Numerical"]["skew"], results["Analytical"]["skew"],
            injected_skew,
        ),
        "orm_metrics": comparison_metrics(
            results["Numerical"]["orm"], results["Analytical"]["orm"]
        ),
        "dispersion_metrics": comparison_metrics(
            results["Numerical"]["dispersion"],
            results["Analytical"]["dispersion"],
        ),
        "numerical_orm_residual_rms": float(
            np.sqrt(np.mean((results["Numerical"]["orm"] - measured[:, :-1])**2))
        ),
        "analytical_orm_residual_rms": float(
            np.sqrt(np.mean((results["Analytical"]["orm"] - measured[:, :-1])**2))
        ),
        "numerical_dispersion_residual_rms": float(
            np.sqrt(np.mean((results["Numerical"]["dispersion"] - measured[:, -1])**2))
        ),
        "analytical_dispersion_residual_rms": float(
            np.sqrt(np.mean((results["Analytical"]["dispersion"] - measured[:, -1])**2))
        ),
        "numerical_jacobian_seconds": results["Numerical"]["jacobian_seconds"].tolist(),
        "analytical_jacobian_seconds": results["Analytical"]["jacobian_seconds"].tolist(),
        "jacobian_speedup": float(
            np.sum(results["Numerical"]["jacobian_seconds"])
            / np.sum(results["Analytical"]["jacobian_seconds"])
        ),
    }
    print("normal+skew comparison summary", summary)
