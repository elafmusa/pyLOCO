from __future__ import annotations

import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from pyLOCO.gui.backend import LocoRunRequest
from pyLOCO.gui.lattice_loader import (
    MultipleMatLatticesError, NoMatLatticeError, load_project_lattice,
    mat_lattice_candidates, select_mat_lattice,
)
from pyLOCO.gui.models.project import LatticeSelection, ProjectMetadata
from pyLOCO.gui.main_window import MainWindow
from pyLOCO.gui.fit_workflow import FitRunSession


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _ring(at, count, prefix):
    return at.Lattice([at.Drift(f"{prefix}{index}", 0.1) for index in range(count)], energy=6.0e9)


def _multi_mat(tmp_path, variables):
    import at
    import scipy.io

    payload = {}
    for name, ring in variables.items():
        single = tmp_path / f"{name}.mat"
        at.save_mat(ring, single, use=name)
        payload[name] = scipy.io.loadmat(single)[name]
    payload["errtab"] = [[1.0, 2.0]]
    payload["cortab"] = [[3.0, 4.0]]
    target = tmp_path / "models.mat"
    scipy.io.savemat(target, payload, long_field_names=True)
    return target


def test_unique_mat_lattice_is_detected_by_structure(tmp_path):
    at = pytest.importorskip("at")
    path = _multi_mat(tmp_path, {"machine_model": _ring(at, 3, "D")})
    candidates = mat_lattice_candidates(path)
    assert [(candidate.variable, candidate.element_count) for candidate in candidates] == [("machine_model", 3)]
    lattice, selected = select_mat_lattice(path)
    assert len(lattice) == 3 and selected.variable == "machine_model"


def test_ring_and_arbitrary_valid_names_are_not_hardcoded(tmp_path):
    at = pytest.importorskip("at")
    path = _multi_mat(tmp_path, {"RING": _ring(at, 2, "R"), "another_valid_lattice": _ring(at, 4, "A")})
    candidates = mat_lattice_candidates(path)
    assert {candidate.variable: candidate.element_count for candidate in candidates} == {
        "RING": 2, "another_valid_lattice": 4,
    }
    with pytest.raises(MultipleMatLatticesError):
        select_mat_lattice(path)
    lattice, selected = select_mat_lattice(path, "another_valid_lattice")
    assert len(lattice) == 4 and selected.variable == "another_valid_lattice"


def test_no_valid_mat_lattice_is_rejected(tmp_path):
    scipy = pytest.importorskip("scipy.io")
    path = tmp_path / "tables-only.mat"
    scipy.savemat(path, {"errtab": [[1.0]], "cortab": [[2.0]]})
    assert mat_lattice_candidates(path) == []
    with pytest.raises(NoMatLatticeError, match="no variable"):
        select_mat_lattice(path)


def test_saved_variable_round_trip_and_backend_request(tmp_path):
    at = pytest.importorskip("at")
    path = _multi_mat(tmp_path, {"first": _ring(at, 2, "F"), "second": _ring(at, 5, "S")})
    project = ProjectMetadata(name="MAT selection")
    project.lattice = LatticeSelection(str(path), "mat", 5, "second")
    restored = ProjectMetadata.load(project.save(tmp_path / "selection.pyloco.json"))
    assert restored.lattice.mat_variable == "second"
    assert len(load_project_lattice(path, restored.lattice.mat_variable)) == 5
    request = LocoRunRequest.from_project(restored)
    assert request.lattice_mat_variable == "second"
    session_path = FitRunSession({}, str(path), "checksum", {}, {}, lattice_mat_variable="second").save(
        tmp_path / "fit-session.json"
    )
    assert FitRunSession.load(session_path).lattice_mat_variable == "second"


def test_legacy_project_with_unique_mat_variable_is_upgraded_in_memory(app, tmp_path, monkeypatch):
    at = pytest.importorskip("at")
    path = _multi_mat(tmp_path, {"legacy_ring": _ring(at, 2, "L")})
    monkeypatch.setattr(MainWindow, "_refresh_reference_model_info", lambda _self: None)
    window = MainWindow()
    window.project.lattice = LatticeSelection(str(path), "mat", None, "")
    window._refresh_ui("")
    assert window.project.lattice.mat_variable == "legacy_ring"
    assert window.lattice_variable.text() == "legacy_ring"
    window.project.modified = False; window.close()


def test_gui_auto_selects_unique_variable_and_displays_it(app, tmp_path, monkeypatch):
    at = pytest.importorskip("at")
    path = _multi_mat(tmp_path, {"unique_model": _ring(at, 3, "U")})
    monkeypatch.setattr(
        "pyLOCO.gui.main_window.QFileDialog.getOpenFileName",
        lambda *_args, **_kwargs: (str(path), ""),
    )
    monkeypatch.setattr(MainWindow, "_refresh_reference_model_info", lambda _self: None)
    window = MainWindow(); window.select_lattice()
    assert window.project.lattice.mat_variable == "unique_model"
    assert window.project.lattice.element_count == 3
    assert window.lattice_variable.text() == "unique_model"
    assert window.lattice_elements.text() == "3"
    assert window.change_mat_variable_button.isEnabled()
    window.project.modified = False; window.close()


def test_gui_prompts_for_multiple_variables_and_can_change_selection(app, tmp_path, monkeypatch):
    at = pytest.importorskip("at")
    path = _multi_mat(tmp_path, {"first": _ring(at, 2, "F"), "second": _ring(at, 5, "S")})
    monkeypatch.setattr(
        "pyLOCO.gui.main_window.QFileDialog.getOpenFileName",
        lambda *_args, **_kwargs: (str(path), ""),
    )
    choices = iter([("second — 5 elements", True), ("first — 2 elements", True)])
    monkeypatch.setattr(
        "pyLOCO.gui.main_window.QInputDialog.getItem",
        lambda *_args, **_kwargs: next(choices),
    )
    monkeypatch.setattr(MainWindow, "_refresh_reference_model_info", lambda _self: None)
    window = MainWindow(); window.select_lattice()
    assert (window.project.lattice.mat_variable, window.project.lattice.element_count) == ("second", 5)
    window.change_mat_variable()
    assert (window.project.lattice.mat_variable, window.project.lattice.element_count) == ("first", 2)
    window.project.modified = False; window.close()


def test_change_mat_variable_opens_selector_for_single_candidate(app, tmp_path, monkeypatch):
    at = pytest.importorskip("at")
    path = _multi_mat(tmp_path, {"betamodel": _ring(at, 3, "B")})
    monkeypatch.setattr(MainWindow, "_refresh_reference_model_info", lambda _self: None)
    calls = []

    def choose(*args, **kwargs):
        calls.append((args, kwargs))
        return "betamodel — 3 elements", True

    monkeypatch.setattr(
        "pyLOCO.gui.main_window.QInputDialog.getItem", choose,
    )
    window = MainWindow()
    window.project.lattice = LatticeSelection(str(path), "mat", 3, "betamodel")
    window.change_mat_variable()

    assert len(calls) == 1
    assert calls[0][0][1] == "Select MAT lattice variable"
    assert "betamodel — 3 elements" in calls[0][0][3]
    # Confirming the already active variable is display-only.
    assert window.project.lattice.mat_variable == "betamodel"
    assert not window.project.modified
    window.close()


def test_gui_invalid_mat_keeps_active_project_unchanged(app, tmp_path, monkeypatch):
    scipy = pytest.importorskip("scipy.io")
    path = tmp_path / "invalid.mat"; scipy.savemat(path, {"table": [[1.0]]})
    monkeypatch.setattr(
        "pyLOCO.gui.main_window.QFileDialog.getOpenFileName",
        lambda *_args, **_kwargs: (str(path), ""),
    )
    warnings = []
    monkeypatch.setattr(
        "pyLOCO.gui.main_window.QMessageBox.warning",
        lambda *_args: warnings.append(_args[-1]),
    )
    monkeypatch.setattr(MainWindow, "_refresh_reference_model_info", lambda _self: None)
    window = MainWindow()
    window.project.lattice = LatticeSelection("original.mat", "mat", 10, "original")
    window.select_lattice()
    assert window.project.lattice == LatticeSelection("original.mat", "mat", 10, "original")
    assert warnings and "active project is unchanged" in warnings[0]
    window.project.modified = False; window.close()


CASE_C = Path("Examples/EBS/multistage_fit/data/betamodel.mat")
MODERN_EBS = Path(os.environ.get("PYLOCO_EBS_TEST_ROOT", "/external/ebs-pysc-pyaml")) / "betamodel.mat"


@pytest.mark.skipif(not CASE_C.is_file(), reason="Historical Case C lattice unavailable")
def test_historical_case_c_detects_betamodel():
    candidates = mat_lattice_candidates(CASE_C)
    assert [(candidate.variable, candidate.element_count) for candidate in candidates] == [("betamodel", 4432)]


@pytest.mark.skipif(not MODERN_EBS.is_file(), reason="Modern EBS lattice unavailable")
def test_modern_ebs_ignores_error_and_correction_tables():
    candidates = mat_lattice_candidates(MODERN_EBS)
    assert [(candidate.variable, candidate.element_count) for candidate in candidates] == [("betamodel", 4434)]
