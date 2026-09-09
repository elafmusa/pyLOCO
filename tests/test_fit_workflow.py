from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from pyLOCO.gui.fit_workflow import (
    FitRecipe, FitRunSession, FitStage, execute_workflow, preflight_recipe,
)


def mapping(label):
    return {
        "LOCOOptions": {"fit_list": [label]}, "RMConfig": {},
        "MachineElements": {}, "FitInitConfig": {"fit_list": [label]},
        "ConstraintConfig": {}, "FixedParameters": {}, "Output": {},
    }


def test_recipe_contains_strategy_but_never_measurement_paths(tmp_path):
    recipe = FitRecipe("two stages", stages=[
        FitStage("Normal optics", mapping("quads"), "original_model"),
        FitStage("Coupling", mapping("skew_quads"), "previous_stage"),
    ])
    target = recipe.save(tmp_path / "recipe.json")
    text = target.read_text()
    assert "measurements" not in text.lower()
    assert [s.name for s in FitRecipe.load(target).stages] == ["Normal optics", "Coupling"]


def test_preflight_rejects_machine_order_and_convention_mismatch(tmp_path):
    lattice = tmp_path / "ring.mat"; lattice.write_bytes(b"ring")
    recipe = FitRecipe(machine_identity="EBS / demo", reference_model_checksum="wrong",
                       data_requirements={"bpm_names": ["B1", "B2"], "orm_convention": "X/Y,H/V"},
                       stages=[FitStage("one", mapping("quads"), "original_model")])
    report = preflight_recipe(recipe, lattice_path=str(lattice),
                              measurement_identity={"machine_profile": "PETRA", "bpm_names": ["B2", "B1"],
                                                    "orm_convention": "other"})
    assert not report["compatible"]
    assert len(report["errors"]) == 4


def test_stable_name_remap_is_explicit_and_unique(tmp_path):
    lattice = tmp_path / "ring.mat"; lattice.write_bytes(b"ring")
    recipe = FitRecipe(element_identities={"normal_quadrupoles": ["Q2", "Q1"]},
                       stages=[FitStage("one", mapping("quads"), "original_model")])
    report = preflight_recipe(recipe, lattice_path=str(lattice), measurement_identity={},
                              lattice_element_names=["Q1", "D", "Q2"])
    assert report["compatible"]
    assert report["remapping"] == {"normal_quadrupoles": [2, 0]}


def test_continuous_and_saved_resume_use_identical_stage2_checkpoint(tmp_path):
    lattice = tmp_path / "ring.mat"; lattice.write_bytes(b"ring")
    base = SimpleNamespace(project_name="test", lattice_path=str(lattice),
                           measurements={"orm": "orm.h5"}, measurement_session={"id": "same"},
                           backend_mapping={})
    recipe = FitRecipe(stages=[FitStage("one", mapping("quads"), "original_model"),
                               FitStage("two", mapping("skew_quads"), "previous_stage")])
    calls = []

    def runner(request, **_):
        root = tmp_path / f"run-{len(calls)}"; root.mkdir()
        for name in ("ring_pyloco.mat", "fit_dict.pkl", "fit_results.npy", "blocks.pkl", "summary.json"):
            (root / name).write_bytes(b"checkpoint")
        calls.append(request.backend_mapping.get("Resume", {}).copy())
        return SimpleNamespace(results_dir=str(root))

    session = execute_workflow(base, recipe, session_path=tmp_path / "session.json", runner=runner)
    assert calls[1]["directory"] == session.checkpoints[0].results_dir
    saved = FitRunSession.load(tmp_path / "session.json")
    assert saved.checkpoints[0].results_dir == calls[1]["directory"]
    assert saved.checkpoints[0].validate_files() == []
    stage1_only = FitRunSession(
        recipe=saved.recipe, lattice_path=saved.lattice_path,
        lattice_checksum=saved.lattice_checksum, measurements=saved.measurements,
        measurement_identity=saved.measurement_identity, checkpoints=[saved.checkpoints[0]],
    )
    stage1_only.save(tmp_path / "stage1-session.json")
    reloaded = FitRunSession.load(tmp_path / "stage1-session.json")
    execute_workflow(base, recipe, session_path=tmp_path / "resumed.json", runner=runner,
                     resume_session=reloaded)
    assert calls[2]["directory"] == calls[1]["directory"]
