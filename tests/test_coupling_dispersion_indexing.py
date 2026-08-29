from __future__ import annotations

import numpy as np
import pytest

from pyLOCO.pyloco import (
    _canonicalize_measured_response,
    _validate_loco_system_dimensions,
    build_iNoCoupling,
    remove_coupling,
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
    np.testing.assert_array_equal(filtered[0][-1572:], arrays[0][-1572:])
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
