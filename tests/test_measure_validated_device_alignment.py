from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pyLOCO.measure.reference_model import ReferenceModel, model_dispersion, model_orm
from pyLOCO.measure.project import MeasureProject, load_measure_project, save_measure_project


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _Ring(list):
    def __init__(self, size=8):
        super().__init__([object() for _ in range(size)])
        self.requested_refpts = None

    def get_optics(self, *, refpts):
        self.requested_refpts = tuple(int(value) for value in refpts)
        dispersion = np.asarray([[float(value), 0.0, float(value + 100), 0.0] for value in refpts])
        return None, None, SimpleNamespace(dispersion=dispersion)


def _model(ring):
    return ReferenceModel(
        path=Path("ring.mat"), checksum_sha256="checksum", source="test", ring=ring,
        tune=(0.1, 0.2), momentum_compaction=1e-4, inverse_gamma_squared=0.0,
        slip_factor=1e-4, nominal_rf_hz=1.0,
    )


def test_validated_bpm_ordinals_preserve_measurement_order_for_dispersion():
    ring = _Ring()
    devices = [SimpleNamespace(name="control/name/first"), SimpleNamespace(name="control/name/second")]
    x, y = model_dispersion(_model(ring), devices, bpm_ordinals=[3, 1])
    assert ring.requested_refpts == (1, 3)
    np.testing.assert_array_equal(x, [3.0, 1.0])
    np.testing.assert_array_equal(y, [103.0, 101.0])


def test_validated_ordinals_preserve_bpm_and_corrector_order_for_orm(monkeypatch):
    ring = _Ring()
    devices = [SimpleNamespace(name="not-an-at-name") for _ in range(5)]
    raw = np.arange(12.0).reshape(4, 3)
    monkeypatch.setattr("pyLOCO.response_matrix.response_matrix", lambda *_args, **_kwargs: raw.copy())
    result = model_orm(
        _model(ring), devices[:2], devices[2:4], devices[4:], [1.0, 2.0], [3.0], scaled=False,
        bpm_ordinals=[3, 1], horizontal_corrector_ordinals=[4, 2], vertical_corrector_ordinals=[5],
    )
    np.testing.assert_array_equal(result, raw[[1, 0, 3, 2]][:, [1, 0, 2]])


def test_validated_ordinals_reject_duplicates_and_out_of_lattice_values():
    ring = _Ring()
    devices = [SimpleNamespace(name="one"), SimpleNamespace(name="two")]
    with pytest.raises(ValueError, match="duplicate AT ordinals"):
        model_dispersion(_model(ring), devices, bpm_ordinals=[1, 1])
    with pytest.raises(ValueError, match="outside the reference lattice"):
        model_dispersion(_model(ring), devices, bpm_ordinals=[1, 99])


def test_pyaml_comparison_is_disabled_until_complete_alignment(qapp):
    from pyLOCO.measure.main_window import MeasureMainWindow, default_mock_devices

    window = MeasureMainWindow(devices=default_mock_devices(2))
    window.adapter_combo.blockSignals(True)
    window.adapter_combo.setCurrentIndex(window.adapter_combo.findData("pyaml"))
    window.adapter_combo.blockSignals(False)
    window.measurement_device_alignment = None
    window._update_reference_comparison_enabled()
    assert not window.compare_reference.isEnabled()
    window.measurement_device_alignment = {
        "mappings": [
            {"role": "bpm", "status": "mapped"},
            {"role": "horizontal_corrector", "status": "unresolved"},
        ]
    }
    window._update_reference_comparison_enabled()
    assert not window.compare_reference.isEnabled()
    window.measurement_device_alignment["mappings"][1]["status"] = "mapped"
    window._update_reference_comparison_enabled()
    assert window.compare_reference.isEnabled()
    window.close()


def test_measure_project_round_trips_selected_mat_variable(tmp_path):
    path = tmp_path / "measure.pyloco-measure.json"
    save_measure_project(path, MeasureProject(reference_model_mat_variable="betamodel"))
    assert load_measure_project(path).reference_model_mat_variable == "betamodel"


def test_measure_prompts_for_multiple_valid_mat_variables(qapp, tmp_path, monkeypatch):
    from pyLOCO.gui.lattice_loader import MatLatticeCandidate, MultipleMatLatticesError
    from pyLOCO.measure.main_window import MeasureMainWindow, default_mock_devices

    source = tmp_path / "models.mat"; source.write_bytes(b"diagnostic")
    selected_model = replace(_model(_Ring(5)), path=source, mat_variable="second")
    calls = []

    def load(_path, *, source, mat_key=None):
        calls.append(mat_key)
        if mat_key is None:
            raise MultipleMatLatticesError([MatLatticeCandidate("first", 2), MatLatticeCandidate("second", 5)])
        return selected_model

    monkeypatch.setattr("pyLOCO.measure.main_window.load_reference_model", load)
    monkeypatch.setattr("pyLOCO.measure.main_window.QInputDialog.getItem", lambda *_args, **_kwargs: ("second — 5 elements", True))
    window = MeasureMainWindow(devices=default_mock_devices(2))
    assert window._load_reference_model(source, "test")
    assert calls == [None, "second"]
    assert window.reference_model_variable.text() == "second"
    assert window.reference_model_element_count.text() == "5"
    assert "<b>Elements:</b> 5" in window.reference_model_summary.text()
    assert window.measurement_device_alignment is None
    window.close()


REPOSITORY_ROOT = Path(__file__).parents[1]
MODERN_EBS = Path(os.environ.get("PYLOCO_EBS_TEST_ROOT", "/external/ebs-pysc-pyaml")) / "betamodel.mat"
HISTORICAL_EBS = REPOSITORY_ROOT / "Examples/EBS/multistage_fit/data/betamodel.mat"


@pytest.mark.skipif(not MODERN_EBS.is_file(), reason="Modern EBS lattice unavailable")
def test_measure_loads_modern_ebs_betamodel_4434():
    from pyLOCO.measure.reference_model import load_reference_model
    model = load_reference_model(MODERN_EBS)
    assert (model.mat_variable, model.element_count) == ("betamodel", 4434)


@pytest.mark.skipif(not HISTORICAL_EBS.is_file(), reason="Historical EBS lattice unavailable")
def test_measure_loads_historical_ebs_betamodel_4432():
    from pyLOCO.measure.reference_model import load_reference_model
    model = load_reference_model(HISTORICAL_EBS)
    assert (model.mat_variable, model.element_count) == ("betamodel", 4432)


@pytest.mark.skipif(not HISTORICAL_EBS.is_file(), reason="Historical EBS lattice unavailable")
def test_measure_loads_unambiguous_ring_variable(tmp_path):
    scipy = pytest.importorskip("scipy.io")
    from pyLOCO.measure.reference_model import load_reference_model

    payload = scipy.loadmat(HISTORICAL_EBS)["betamodel"]
    path = tmp_path / "ring.mat"
    scipy.savemat(path, {"RING": payload}, long_field_names=True)
    model = load_reference_model(path)
    assert (model.mat_variable, model.element_count) == ("RING", 4432)
