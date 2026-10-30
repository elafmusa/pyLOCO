import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from pyLOCO.gui.backend import LocoRunRequest, _make_results_dir
from pyLOCO.gui.fit_workflow import FitRecipe, FitStage
from pyLOCO.gui.main_window import MainWindow
from pyLOCO.gui.models.project import CompletedRunReference, ImportedDataset, LatticeSelection


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _completed_run(window: MainWindow, root: Path, name: str = "completed") -> Path:
    lattice = root / "ring.mat"
    lattice.write_bytes(b"same lattice")
    orm = root / "orm.h5"
    orm.write_bytes(b"ordered measurement identities")
    window.project.lattice = LatticeSelection(path=str(lattice), file_type="mat", mat_variable="betamodel")
    window.project.measurements = {
        "orm": ImportedDataset(role="orm", path=str(orm), file_type="h5", size_bytes=orm.stat().st_size)
    }
    elements = window.project.loco_config.machine_elements
    elements.bpm_ords = [7, 3]
    elements.horizontal_corrector_ords = [10]
    elements.vertical_corrector_ords = [12]
    request = LocoRunRequest.from_project(window.project)
    result = root / name
    result.mkdir()
    (result / "summary.json").write_text(json.dumps({"completed_utc": "2026-10-24T00:00:00Z"}))
    (result / "run_request.json").write_text(json.dumps({
        "lattice_path": request.lattice_path,
        "lattice_mat_variable": request.lattice_mat_variable,
        "measurements": request.measurements,
        "backend_mapping": request.backend_mapping,
        "input_fingerprints": {
            "lattice_sha256": __import__("hashlib").sha256(lattice.read_bytes()).hexdigest(),
            "measurement_sha256": {
                "orm": __import__("hashlib").sha256(orm.read_bytes()).hexdigest(),
            },
        },
    }))
    for required in ("ring_pyloco.mat", "fit_dict.pkl", "fit_results.npy"):
        (result / required).write_bytes(b"saved state")
    window.project.completed_run = CompletedRunReference(str(result), 1.0, "completed")
    return result


def test_latest_completed_fit_is_compatible_and_button_uses_it(app, tmp_path, monkeypatch):
    window = MainWindow()
    result = _completed_run(window, tmp_path)
    candidate, reason = window._latest_compatible_completed_run()
    assert candidate == result.resolve()
    assert reason == ""

    selected = []
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.Yes)
    monkeypatch.setattr(window, "_start_loco_from", lambda path: selected.append(path))
    window.continue_last_fit()
    assert selected == [result.resolve()]
    window.project.modified = False
    window.close()


def test_continuation_rejects_changed_measurement_and_incomplete_state(app, tmp_path):
    window = MainWindow()
    result = _completed_run(window, tmp_path)
    Path(window.project.measurements["orm"].path).write_bytes(b"different measurement")
    assert "content differs" in window._continuation_compatibility_error(result)
    Path(window.project.measurements["orm"].path).write_bytes(b"ordered measurement identities")
    (result / "fit_dict.pkl").unlink()
    assert "fit_dict.pkl" in window._continuation_compatibility_error(result)
    window.project.modified = False
    window.close()


def test_continuation_uses_resume_for_single_and_first_stage_for_multistage(app, tmp_path):
    window = MainWindow()
    request = LocoRunRequest(
        project_name="test", project_path="", lattice_path="ring.mat",
        measurements={}, backend_mapping={"Resume": {"enabled": False}},
    )
    result = tmp_path / "previous"
    assert window._apply_continuation_to_request(request, None, result) is None
    assert request.backend_mapping["Resume"]["directory"] == str(result.resolve())

    recipe = FitRecipe(stages=[
        FitStage("Stage 1", {}, "original_model"),
        FitStage("Stage 2", {}, "previous_stage"),
    ])
    continued = window._apply_continuation_to_request(request, recipe, result)
    assert recipe.stages[0].start_from == "original_model"
    assert continued.stages[0].start_from == "saved_result"
    assert continued.stages[0].saved_result == str(result)
    assert continued.stages[1].start_from == "previous_stage"

    request.backend_mapping["Resume"] = {"enabled": True, "directory": "old"}
    fresh = window._apply_fresh_start_to_request(request, continued)
    assert request.backend_mapping["Resume"] == {"enabled": False}
    assert fresh.stages[0].start_from == "original_model"
    assert fresh.stages[0].saved_result == ""
    assert continued.stages[0].start_from == "saved_result"
    window.project.modified = False
    window.close()


def test_result_directory_never_reuses_existing_run(tmp_path):
    request = LocoRunRequest(
        project_name="repeat", project_path="", lattice_path="ring.mat",
        measurements={}, backend_mapping={"Output": {"directory": str(tmp_path), "run_name": "same"}},
    )
    first = _make_results_dir(request)
    second = _make_results_dir(request)
    assert first != second
    assert first.is_dir() and second.is_dir()
