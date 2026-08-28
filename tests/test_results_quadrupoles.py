from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pyLOCO.gui.results.parameters_view import ParametersView
from pyLOCO.gui.results.results_loader import ResultsLoader


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _results(tmp_path, values, ordinals, *, individuals):
    (tmp_path / "summary.json").write_text(json.dumps({
        "blocks": {"quads": {"start": 0, "stop": len(values)}},
        "chi2_history": [1.0],
    }), encoding="utf-8")
    (tmp_path / "run_request.json").write_text(json.dumps({
        "lattice_path": "initial.mat",
        "backend_mapping": {
            "LOCOOptions": {"fit_list": ["quads"]},
            "MachineElements": {"normal_quadrupole_ords": ordinals},
            "FitInitConfig": {"individuals": individuals, "quads_attr": "PolynomB", "quads_attr_index": 1},
        },
    }), encoding="utf-8")
    np.savez_compressed(tmp_path / "loco_results.npz", fit_results=np.asarray([values], dtype=float))


def _fake_at(monkeypatch, families, strengths):
    ring = [SimpleNamespace(FamName=name, PolynomB=np.asarray([0.0, strength])) for name, strength in zip(families, strengths)]
    monkeypatch.setitem(sys.modules, "at", SimpleNamespace(load_lattice=lambda _path: ring))


def test_quadrupole_plot_and_table_share_exact_individual_order(app, tmp_path, monkeypatch):
    _results(tmp_path, [1.1, 1.8, 3.6], [0, 1, 2], individuals=True)
    _fake_at(monkeypatch, ["Q1", "Q2", "Q3"], [1.0, 2.0, 3.0])
    loader = ResultsLoader(tmp_path)
    rows = loader.quadrupole_parameter_rows
    assert [row["name"] for row in rows] == ["Q1", "Q2", "Q3"]
    assert [row["lattice_ordinal"] for row in rows] == [0, 1, 2]
    np.testing.assert_allclose([row["delta_k"] for row in rows], [.1, -.2, .6])

    view = ParametersView(); view.set_loader(loader)
    view.selector.setCurrentIndex(view.selector.findData("__quad_delta__")); app.processEvents()
    assert view.table.rowCount() == len(rows) == 3
    np.testing.assert_allclose(view.plot.figure.axes[0].lines[0].get_ydata(), [.1, -.2, .6])
    assert view.table.horizontalHeaderItem(6).text() == "ΔK/K [%]"
    assert view.plot.save_button.isEnabled()
    view.selector.setCurrentIndex(view.selector.findData("__quad_relative__")); app.processEvents()
    np.testing.assert_allclose(view.plot.figure.axes[0].lines[0].get_ydata(), [10.0, -10.0, 20.0])
    view.close()


def test_family_fit_has_one_named_row_per_family_without_fake_ordinal(app, tmp_path, monkeypatch):
    _results(tmp_path, [1.2, 2.1], [0, 1, 2, 3], individuals=False)
    _fake_at(monkeypatch, ["QF", "QF", "QD", "QD"], [1.0, 1.0, 2.0, 2.0])
    loader = ResultsLoader(tmp_path)
    rows = loader.quadrupole_parameter_rows
    assert [row["name"] for row in rows] == ["QF", "QD"]
    assert [row["member_ordinals"] for row in rows] == [[0, 1], [2, 3]]
    assert all(row["lattice_ordinal"] is None for row in rows)
    view = ParametersView(); view.set_loader(loader)
    view.selector.setCurrentIndex(view.selector.findData("__quad_delta__")); app.processEvents()
    assert view.table.rowCount() == 2
    assert view.table.item(0, 1).text() == "QF"
    assert view.table.item(0, 2).text() == "—"
    assert "0, 1" in view.table.item(0, 1).toolTip()
    view.close()


def test_reopened_loader_reproduces_quadrupole_rows(app, tmp_path, monkeypatch):
    _results(tmp_path, [1.05, 1.9], [0, 1], individuals=True)
    _fake_at(monkeypatch, ["QA", "QB"], [1.0, 2.0])
    first = ResultsLoader(tmp_path).quadrupole_parameter_rows
    second_loader = ResultsLoader(tmp_path)
    second = second_loader.quadrupole_parameter_rows
    assert second == first
    view = ParametersView(); view.set_loader(second_loader)
    view.selector.setCurrentIndex(view.selector.findData("__quad_delta__")); app.processEvents()
    assert view.table.rowCount() == len(second) == 2
    view.close()
