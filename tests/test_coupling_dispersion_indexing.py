from __future__ import annotations

import numpy as np
import pytest

from pyLOCO.pyloco import (
    _canonicalize_measured_response,
    _validate_loco_system_dimensions,
    build_orm_term_indices,
    build_iNoCoupling,
    compute_chi_squared_,
    pyloco,
    remove_coupling,
    select_orm_terms,
)


def _vectors(n_hbpm, n_vbpm, n_hcor, n_vcor, include_dispersion):
    size = (n_hbpm + n_vbpm) * (
        n_hcor + n_vcor + int(include_dispersion)
    )
    measured = np.arange(size, dtype=float)[:, None]
    model = measured + 0.5
    weights = np.linspace(1.0, 2.0, size)[:, None]
    jacobian = np.column_stack((measured[:, 0], measured[:, 0] + 1.0))
    return measured, model, weights, jacobian


def _expected_non_coupling_indices(
        n_hbpm, n_vbpm, n_hcor, n_vcor, include_dispersion):
    n_bpm = n_hbpm + n_vbpm
    n_columns = n_hcor + n_vcor + int(include_dispersion)
    index_matrix = np.arange(n_bpm * n_columns).reshape(
        n_bpm, n_columns, order="F"
    )
    pieces = [
        index_matrix[:n_hbpm, :n_hcor].ravel(order="F"),
        index_matrix[n_hbpm:, n_hcor:n_hcor + n_vcor].ravel(order="F"),
    ]
    if include_dispersion:
        pieces.append(index_matrix[:, -1])
    return np.concatenate(pieces)


def test_remove_coupling_without_dispersion_asymmetric_dimensions():
    counts = (5, 3, 4, 2)
    arrays = _vectors(*counts, False)
    filtered = remove_coupling(*arrays, *counts, False)
    expected = _expected_non_coupling_indices(*counts, False)
    np.testing.assert_array_equal(filtered[4], expected)
    np.testing.assert_array_equal(filtered[5], expected)
    for actual, source in zip(filtered[:4], arrays):
        np.testing.assert_array_equal(actual, source[expected])


@pytest.mark.parametrize("mode", ["full", "diagonal", "off_diagonal"])
def test_orm_term_masks_support_unequal_plane_counts_and_keep_dispersion(mode):
    counts = (5, 3, 4, 2)
    arrays = _vectors(*counts, True)
    filtered = select_orm_terms(*arrays, *counts, True, mode)
    indices, chi_indices, _ = build_orm_term_indices(*counts, True, mode)
    np.testing.assert_array_equal(filtered[4], indices)
    np.testing.assert_array_equal(filtered[5], chi_indices)
    for actual, source in zip(filtered[:4], arrays):
        np.testing.assert_array_equal(actual, source[indices])
    n_bpm = counts[0] + counts[1]
    expected_orm = {
        "full": n_bpm * (counts[2] + counts[3]),
        "diagonal": counts[0] * counts[2] + counts[1] * counts[3],
        "off_diagonal": counts[0] * counts[3] + counts[1] * counts[2],
    }[mode]
    assert indices.size == expected_orm + n_bpm
    np.testing.assert_array_equal(indices[-n_bpm:], np.arange(arrays[0].size - n_bpm, arrays[0].size))


@pytest.mark.parametrize("mode,perturbed_block", [
    ("diagonal", "off_diagonal"),
    ("off_diagonal", "diagonal"),
])
def test_excluded_orm_blocks_do_not_contribute_to_chi_squared(mode, perturbed_block):
    counts = (3, 2, 2, 1)
    n_hbpm, n_vbpm, n_hcor, n_vcor = counts
    n_total = (n_hbpm + n_vbpm) * (n_hcor + n_vcor)
    measured = np.zeros((n_total, 1))
    model = np.zeros_like(measured)
    excluded, _, _ = build_orm_term_indices(*counts, False, perturbed_block)
    model[excluded] = 1.0e9
    selected, _, _ = build_orm_term_indices(*counts, False, mode)
    chi2 = compute_chi_squared_(
        measured, model, np.ones_like(measured),
        nHBPM=n_hbpm, nVBPM=n_vbpm, nHorCOR=n_hcor, nVerCOR=n_vcor,
        include_dispersion=False, remove_coupling_=True, iNoCoupling=selected,
        iOutliers=np.asarray([], dtype=int), n_fit_parameters=np.zeros((n_total, 1)),
    )
    assert chi2 == 0.0


def test_full_orm_includes_cross_plane_residuals_in_objective():
    counts = (3, 2, 2, 1)
    n_total = (counts[0] + counts[1]) * (counts[2] + counts[3])
    measured = np.zeros((n_total, 1))
    model = np.zeros_like(measured)
    cross_plane, _, _ = build_orm_term_indices(*counts, False, "off_diagonal")
    model[cross_plane[0]] = 2.0
    full, _, _ = build_orm_term_indices(*counts, False, "full")
    chi2 = compute_chi_squared_(
        measured, model, np.ones_like(measured),
        nHBPM=counts[0], nVBPM=counts[1], nHorCOR=counts[2], nVerCOR=counts[3],
        include_dispersion=False, remove_coupling_=False, iNoCoupling=full,
        iOutliers=np.asarray([], dtype=int), n_fit_parameters=np.zeros((n_total, 1)),
    )
    assert chi2 > 0.0


def test_analytical_response_calculator_rejects_coupled_objectives_early():
    with pytest.raises(ValueError, match="supports only diagonal ORM terms"):
        pyloco(None, response_matrix_calculator="Analytical", orm_terms="off_diagonal")


def test_remove_coupling_with_dispersion_retains_both_eta_planes_petra_iv_size():
    # Seed-111 local debug dimensions: 786 H/V BPMs and 10 H/V correctors.
    counts = (786, 786, 10, 10)
    arrays = _vectors(*counts, True)
    filtered = remove_coupling(*arrays, *counts, True)
    expected = _expected_non_coupling_indices(*counts, True)
    np.testing.assert_array_equal(filtered[4], expected)
    np.testing.assert_array_equal(filtered[5], expected)
    assert expected.size == 2 * 786 * 10 + 2 * 786
    assert expected.max() < arrays[0].size
    # The final nHBPM+nVBPM entries are eta_x followed by eta_y.
    np.testing.assert_array_equal(
        filtered[0][-1572:], arrays[0][-1572:]
    )
    np.testing.assert_array_equal(filtered[1], arrays[1][expected])
    np.testing.assert_array_equal(filtered[2], arrays[2][expected])
    np.testing.assert_array_equal(filtered[3], arrays[3][expected, :])


def test_dispersion_without_coupling_removal_preserves_full_system():
    counts = (7, 5, 4, 3)
    n_hbpm, n_vbpm, n_hcor, n_vcor = counts
    pure = np.arange((n_hbpm + n_vbpm) * (n_hcor + n_vcor), dtype=float)
    pure = pure.reshape(n_hbpm + n_vbpm, n_hcor + n_vcor)
    eta_x = np.linspace(1.0, 2.0, n_hbpm)
    eta_y = np.linspace(3.0, 4.0, n_vbpm)
    canonical = _canonicalize_measured_response(
        pure, eta_x, eta_y,
        nHBPM=n_hbpm, nVBPM=n_vbpm,
        nHorCOR=n_hcor, nVerCOR=n_vcor,
        include_dispersion=True,
    )
    np.testing.assert_array_equal(canonical[:, :-1], pure)
    np.testing.assert_array_equal(
        canonical[:, -1], np.concatenate((eta_x, eta_y))
    )
    full = canonical.reshape(-1, 1, order="F")
    jacobian = np.ones((full.size, 6))
    assert _validate_loco_system_dimensions(
        full, full.copy(), np.ones_like(full), jacobian,
        nHBPM=n_hbpm, nVBPM=n_vbpm,
        nHorCOR=n_hcor, nVerCOR=n_vcor,
        include_dispersion=True,
    ) == full.size


def test_inconsistent_system_fails_before_coupling_indexing():
    counts = (7, 5, 4, 3)
    measured, model, weights, jacobian = _vectors(*counts, True)
    with pytest.raises(ValueError, match=(
        r"Inconsistent LOCO residual/Jacobian dimensions with dispersion: "
        r"measured=.*nHBPM=7.*nVBPM=5.*nHCor=4.*nVCor=3.*"
        r"includeDispersion=True"
    )):
        remove_coupling(
            measured[:-1], model, weights, jacobian, *counts, True
        )


def test_legacy_appended_response_and_gui_order_match_canonical_interface():
    from pyLOCO.gui.backend import _assemble_measured_response

    measured = {
        "orm": np.arange(60.0).reshape(10, 6),
        "eta_x": np.arange(6.0),
        "eta_y": np.arange(4.0) + 10.0,
    }
    gui_response = _assemble_measured_response(measured, True)
    core_response = _canonicalize_measured_response(
        measured["orm"], measured["eta_x"], measured["eta_y"],
        nHBPM=6, nVBPM=4, nHorCOR=4, nVerCOR=2,
        include_dispersion=True,
    )
    np.testing.assert_array_equal(core_response, gui_response)
    # Already-appended legacy callers remain accepted if their eta agrees.
    np.testing.assert_array_equal(
        _canonicalize_measured_response(
            gui_response, measured["eta_x"], measured["eta_y"],
            nHBPM=6, nVBPM=4, nHorCOR=4, nVerCOR=2,
            include_dispersion=True,
        ),
        gui_response,
    )


def test_build_mask_never_exceeds_canonical_response_length():
    counts = (11, 8, 5, 3)
    fit, chi, n_bpm = build_iNoCoupling(*counts, True)
    full_length = n_bpm * (counts[2] + counts[3] + 1)
    assert fit.max() < full_length
    assert chi.max() < full_length
    np.testing.assert_array_equal(fit, chi)
