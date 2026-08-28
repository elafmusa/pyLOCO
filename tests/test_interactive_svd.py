from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog

from pyLOCO.gui.main_window import SVDSelectionDialog
from pyLOCO.pyloco import _svd_select_indices, solve_step_gn


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_interactive_callback_controls_small_gn_solver_selection():
    # Minimal deterministic analogue of the single-quadrupole reconstruction:
    # three fitted knobs and a deliberately chosen two-mode SVD solution.
    jacobian = np.diag([4.0, 2.0, 1.0])
    residual = np.array([8.0, 4.0, 3.0])
    calls = []

    def choose(singular_values, tag):
        calls.append((singular_values.copy(), tag))
        return [0, 2]

    result, selected, singular_values = solve_step_gn(
        jacobian,
        residual,
        "interactive",
        1e-7,
        None,
        False,
        "single-quad-test",
        np.ones(3),
        np.zeros(3),
        residual,
        svd_selection_callback=choose,
    )

    assert len(calls) == 1
    assert calls[0][1] == "single-quad-test"
    np.testing.assert_allclose(calls[0][0], singular_values)
    np.testing.assert_array_equal(selected, [0, 2])
    np.testing.assert_allclose(result, [2.0, 0.0, 3.0])


@pytest.mark.parametrize("selection", [[], [-1], [3], [0.5]])
def test_interactive_callback_rejects_invalid_selection(selection):
    values = np.array([3.0, 2.0, 1.0])
    with pytest.raises((ValueError, TypeError)):
        _svd_select_indices(
            values,
            np.eye(3),
            np.eye(3),
            np.ones(3),
            np.eye(3),
            np.ones(3),
            np.zeros(3),
            np.ones(3),
            method="interactive",
            selection_callback=lambda _values, _tag: selection,
        )


def test_qt_dialog_returns_checked_singular_values(app):
    dialog = SVDSelectionDialog([10.0, 1.0, 0.1], "GN it1", 2)
    assert dialog.selected_indices() == [0, 1]
    dialog.table.item(1, 0).setCheckState(Qt.Unchecked)
    assert dialog.selected_indices() == [0]
    dialog._accept_if_valid()
    assert dialog.result() == QDialog.Accepted
