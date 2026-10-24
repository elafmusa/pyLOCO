from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pyLOCO.gui.backend import LocoRunRequest
from pyLOCO.gui.fit_workflow import FitRecipe, FitRunSession, execute_workflow
from pyLOCO.gui.models.project import ProjectMetadata


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "Examples" / "EBS" / "mdt_multistage"
PROJECT = EXAMPLE / "EBS_MDT_20261024.pyloco.json"
RECIPE = EXAMPLE / "recipe.json"
SPEC = importlib.util.spec_from_file_location("ebs_mdt_multistage", EXAMPLE / "run_ebs_mdt_multistage.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_project_and_four_stage_recipe_use_current_mdt_inventory():
    project = ProjectMetadata.load(PROJECT)
    recipe = FitRecipe.load(RECIPE)
    request = LocoRunRequest.from_project(project)

    assert len(recipe.stages) == 4
    assert [stage.start_from for stage in recipe.stages] == [
        "original_model", "previous_stage", "previous_stage", "previous_stage"
    ]
    assert [stage.name for stage in recipe.stages] == [
        "BPM gains and corrector calibration", "BPM coupling",
        "Normal quadrupoles", "Skew quadrupoles and coupling",
    ]
    expected_fit_lists = [
        ["hbpm_gain", "vbpm_gain", "hcor_cal", "vcor_cal"],
        ["hbpm_coupling", "vbpm_coupling"],
        ["quads"],
        ["skew_quads", "hbpm_coupling", "vbpm_coupling", "hcor_coupling", "vcor_coupling"],
    ]
    for stage, expected in zip(recipe.stages, expected_fit_lists):
        mapping = stage.configuration["backend_mapping"]
        assert mapping["FitInitConfig"]["fit_list"] == expected
        assert mapping["RMConfig"]["Frequency"] == 352372169.3993786
        assert mapping["RMConfig"]["HarmNumber"] == 992
        assert mapping["MachineElements"]["bpm_ords"] == request.backend_mapping["MachineElements"]["bpm_ords"]
        assert mapping["MachineElements"]["horizontal_corrector_ords"] == [30, 117, 191, 6, 289]
        assert mapping["MachineElements"]["vertical_corrector_ords"] == [30, 117, 191, 6, 289]
    assert len(recipe.stages[2].configuration["backend_mapping"]["MachineElements"]["normal_quadrupole_ords"]) == 32
    assert len(recipe.stages[3].configuration["backend_mapping"]["MachineElements"]["skew_quadrupole_ords"]) == 288
    assert "/Users/" not in PROJECT.read_text(encoding="utf-8")


def test_scientific_preflight_validates_exact_saved_payload():
    project = ProjectMetadata.load(PROJECT)
    recipe = FitRecipe.load(RECIPE)
    report = MODULE.scientific_preflight(project, recipe)

    assert report["compatible"], report["errors"]
    assert report["inventory"] == {
        "bpms": 320, "horizontal_correctors": 5, "vertical_correctors": 5
    }
    assert report["lattice"]["element_count"] == 4434
    assert report["lattice"]["rf_frequency_hz"] == 352372169.3993786
    assert report["lattice"]["harmonic_number"] == 992
    dispersion = report["measurements"]["files"]["dispersion"]
    assert dispersion["rf_step_hz"] == 200.0
    assert dispersion["source_rf_difference_sign_convention"] == "negative_minus_positive"
    assert dispersion["rf_difference_sign_convention"] == "positive_minus_negative"


def test_smoke_recipe_only_shortens_iteration_counts():
    recipe = FitRecipe.load(RECIPE)
    smoke = MODULE.smoke_recipe(recipe)
    assert [stage.name for stage in smoke.stages] == [stage.name for stage in recipe.stages]
    for original, shortened in zip(recipe.stages, smoke.stages):
        before = original.configuration["backend_mapping"]
        after = shortened.configuration["backend_mapping"]
        assert after["FitInitConfig"] == before["FitInitConfig"]
        assert after["MachineElements"] == before["MachineElements"]
        assert after["LOCOOptions"]["nIter"] == 1
        assert after["LOCOOptions"]["nLMIter"] == 1


def _write_mock_stage(root: Path, stage_number: int) -> SimpleNamespace:
    root.mkdir(parents=True)
    for name in ("ring_pyloco.mat", "fit_dict.pkl", "fit_results.npy", "blocks.pkl"):
        (root / name).write_bytes(b"test")
    (root / "summary.json").write_text(json.dumps({
        "chi2_history": [10.0 / stage_number],
        "iteration_metrics": [{"orm_residual": {"rms": 1e-4 / stage_number}}],
    }))
    np.savez_compressed(
        root / "optics_results.npz",
        beta_beating_x=np.array([0.1 / stage_number]),
        beta_beating_y=np.array([0.2 / stage_number]),
    )
    np.savez_compressed(
        root / "loco_results.npz",
        fit_results=np.array([[0.0, 0.0], [stage_number, -stage_number]], dtype=float),
        singular_values=np.array([2.0, 1.0]),
    )
    return SimpleNamespace(results_dir=str(root))


def test_continuation_persistence_reopening_and_comparison(tmp_path, monkeypatch):
    project = ProjectMetadata.load(PROJECT)
    recipe = FitRecipe.load(RECIPE)
    request = LocoRunRequest.from_project(project)
    session_path = tmp_path / "workflow.json"
    resume_directories = []
    stage_number = 0

    monkeypatch.setattr("pyLOCO.gui.fit_workflow._rebase_workflow_optics", lambda *a, **k: True)

    def runner(stage_request, **_callbacks):
        nonlocal stage_number
        stage_number += 1
        resume_directories.append(stage_request.backend_mapping["Resume"].get("directory"))
        return _write_mock_stage(tmp_path / f"stage-{stage_number}", stage_number)

    session = execute_workflow(request, recipe, session_path=session_path, runner=runner)
    assert len(session.checkpoints) == 4
    assert resume_directories[0] in (None, "")
    assert resume_directories[1:] == [checkpoint.results_dir for checkpoint in session.checkpoints[:-1]]
    reopened = FitRunSession.load(session_path)
    assert MODULE.validate_resume_session(reopened, request, recipe) == []
    assert [item.stage_name for item in reopened.checkpoints] == [stage.name for stage in recipe.stages]
    rows = MODULE.stage_comparison(reopened)
    assert [row["stage"] for row in rows] == [1, 2, 3, 4]
    assert all(row["svd_available"] for row in rows)
    assert rows[-1]["chi2"] == 2.5

    called = False
    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("Completed stages must not rerun")
    resumed = execute_workflow(
        request, recipe, session_path=session_path, runner=forbidden_runner,
        resume_session=reopened,
    )
    assert not called
    assert len(resumed.checkpoints) == 4

    changed = MODULE.smoke_recipe(recipe)
    assert "different FIT recipe" in MODULE.validate_resume_session(
        reopened, request, changed
    )[0]
