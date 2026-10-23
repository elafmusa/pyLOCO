import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from pyLOCO.gui.main_window import ElementSelectionDialog, MainWindow
from pyLOCO.gui.models.project import ProjectMetadata


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class Quadrupole:
    def __init__(self, name: str):
        self.FamName = name
        self.PolynomB = np.asarray([0.0, 1.0])


def _quad_dialog(app, count=12):
    lattice = [Quadrupole(f"QD3J-{index:02d}" if index < 6 else f"QF4J-{index:02d}") for index in range(count)]

    class Parent(QWidget):
        def _load_current_lattice(self):
            return lattice

    parent = Parent()
    dialog = ElementSelectionDialog(parent, "normal_quadrupole_ords", [])
    dialog.type_radio.setChecked(True)
    assert dialog._preview()
    return parent, dialog


def test_comma_wildcard_filters_and_small_n_controls(app):
    parent, dialog = _quad_dialog(app)
    dialog.search_edit.setText("QD3J-00, QF4J-07")
    assert dialog._visible_ordinals() == [0, 7]
    dialog.search_edit.setText("QD3*,")
    assert dialog._visible_ordinals() == list(range(6))

    dialog.selection_n.setValue(3)
    dialog._select_first_n()
    assert dialog._checked_ordinals() == [0, 1, 2]
    dialog.selection_n.setValue(5)
    dialog.search_edit.clear()
    dialog._select_uniform_n()
    assert dialog._checked_ordinals() == [0, 2, 5, 8, 11]
    assert "5 selected / 12 available" in dialog.selection_count.text()
    dialog.close(); parent.close()


def test_highlight_checkbox_filter_and_sort_keep_independent_order(app):
    parent, dialog = _quad_dialog(app, 8)
    dialog._clear_all()
    selection = dialog.table.selectionModel()
    flags = QItemSelectionModel.Select | QItemSelectionModel.Rows
    rows = {
        int(dialog.table.item(row, 0).data(Qt.UserRole)): row
        for row in range(dialog.table.rowCount())
    }
    selection.select(dialog.table.model().index(rows[1], 0), flags)
    selection.select(dialog.table.model().index(rows[3], 0), flags)
    dialog._select_only_highlighted()
    assert dialog._checked_ordinals() == [1, 3]

    dialog.table.clearSelection()
    rows = {
        int(dialog.table.item(row, 0).data(Qt.UserRole)): row
        for row in range(dialog.table.rowCount())
    }
    selection.select(dialog.table.model().index(rows[5], 0), flags)
    dialog._add_highlighted()
    assert dialog._checked_ordinals() == [1, 3, 5]
    dialog.search_edit.setText("QD3J")
    dialog.table.sortItems(2, Qt.DescendingOrder)
    assert dialog._checked_ordinals() == [1, 3, 5]
    target_row = next(
        row for row in range(dialog.table.rowCount())
        if int(dialog.table.item(row, 0).data(Qt.UserRole)) == 3
    )
    dialog.table.item(target_row, 0).setCheckState(Qt.Unchecked)
    assert dialog._checked_ordinals() == [1, 5]
    dialog.close(); parent.close()


def test_copy_normal_quadrupoles_to_tilts_is_independent_and_persistent(app, monkeypatch, tmp_path):
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: None)
    window = MainWindow()
    lattice = [Quadrupole("QD3J"), Quadrupole("QF4J"), Quadrupole("QD5B")]
    monkeypatch.setattr(window, "_load_current_lattice", lambda: lattice)
    elements = window.project.loco_config.machine_elements
    elements.normal_quadrupole_ords = [0, 1, 2]
    elements.quadrupole_tilt_ords = [0]

    window.copy_normal_quadrupoles_to_tilts(mode="replace")
    assert elements.quadrupole_tilt_ords == [0, 1, 2]
    state = window.project.loco_config.element_selection_state["quadrupole_tilt_ords"]
    assert state["method"] == "copied_normal_quadrupoles"
    assert state["resolved_ords"] == [0, 1, 2]
    saved = window.project.save(tmp_path / "tilt-copy.pyloco.json")
    restored = ProjectMetadata.load(saved)
    assert restored.loco_config.machine_elements.quadrupole_tilt_ords == [0, 1, 2]
    assert restored.loco_config.element_selection_state["quadrupole_tilt_ords"]["method"] == "copied_normal_quadrupoles"

    elements.normal_quadrupole_ords = [2]
    assert elements.quadrupole_tilt_ords == [0, 1, 2]
    reopened = ElementSelectionDialog(window, "quadrupole_tilt_ords", [0, 1, 2], state)
    assert reopened.selected_ords == [0, 1, 2]
    assert "Copied from Normal Quadrupoles" in reopened.provenance_summary.text()
    reopened.close(); window.project.modified = False; window.close()
