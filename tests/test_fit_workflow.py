from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

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


def _modern_measurements(tmp_path, *, bpm_names=("B1", "B2"), h_names=("H1",),
                         v_names=("V1",), columns=None):
    paths = {}
    with h5py.File(tmp_path / "orm.h5", "w") as handle:
        names = handle.create_group("names")
        text = h5py.string_dtype("utf-8")
        names.create_dataset("bpms", data=list(bpm_names), dtype=text)
        names.create_dataset("horizontal_correctors", data=list(h_names), dtype=text)
        names.create_dataset("vertical_correctors", data=list(v_names), dtype=text)
        handle.create_dataset("response_matrix", data=np.zeros((2 * len(bpm_names), columns or len(h_names) + len(v_names))))
        handle.attrs["response_matrix_unit"] = "m"
        handle.attrs["row_order"] = "horizontal_bpms,vertical_bpms"
        handle.attrs["column_order"] = "horizontal_correctors,vertical_correctors"
    paths["orm"] = str(tmp_path / "orm.h5")
    for role, datasets in (("dispersion", ("measured_eta_x", "measured_eta_y")),
                           ("bpm_noise", ("Noise_BPMx", "Noise_BPMy"))):
        path = tmp_path / f"{role}.h5"
        with h5py.File(path, "w") as handle:
            names = handle.create_group("names")
            names.create_dataset("bpms", data=list(bpm_names), dtype=h5py.string_dtype("utf-8"))
            for name in datasets:
                handle.create_dataset(name, data=np.zeros(len(bpm_names)))
            if role == "dispersion":
                handle.attrs["measured_eta_unit"] = "m"
                handle.attrs["rf_bipolar_separation_hz"] = 200.0
            else:
                metadata = handle.create_group("metadata")
                metadata.create_dataset("json", data=json.dumps({"bpm_noise_unit": "m"}))
        paths[role] = str(path)
    return paths


def test_preflight_reads_selected_hdf5_payload_not_stale_session_identity(tmp_path):
    lattice = tmp_path / "ring.mat"; lattice.write_bytes(b"ring")
    requirements = {
        "bpm_names": ["B1", "B2"], "horizontal_corrector_names": ["H1"],
        "vertical_corrector_names": ["V1"], "rf_step_hz": 200.0,
        "units": {"orm": "m", "dispersion": "m", "bpm_noise": "m"},
    }
    recipe = FitRecipe(data_requirements=requirements,
                       stages=[FitStage("one", mapping("quads"), "original_model")])
    valid = _modern_measurements(tmp_path)
    stale = {**requirements, "machine_profile": ""}
    report = preflight_recipe(recipe, lattice_path=str(lattice), measurement_identity=stale,
                              measurements=valid)
    assert report["compatible"]
    assert report["measurement_payload"]["files"]["orm"]["shape"] == [4, 2]

    invalid_root = tmp_path / "replacement"; invalid_root.mkdir()
    invalid = _modern_measurements(invalid_root, h_names=("OTHER",), v_names=(), columns=1)
    report = preflight_recipe(recipe, lattice_path=str(lattice), measurement_identity=stale,
                              measurements=invalid)
    assert not report["compatible"]
    assert any("Selected ORM shape (4, 1)" in error for error in report["errors"])
    assert any("horizontal_corrector_names" in error for error in report["errors"])


def test_preflight_rejects_stage_selection_incompatible_with_payload(tmp_path):
    lattice = tmp_path / "ring.mat"; lattice.write_bytes(b"ring")
    paths = _modern_measurements(tmp_path)
    stage_mapping = mapping("quads")
    stage_mapping["MachineElements"] = {
        "bpm_ords": [1, 2], "horizontal_corrector_ords": [3, 4],
        "vertical_corrector_ords": [5], "normal_quadrupole_ords": [],
    }
    recipe = FitRecipe(data_requirements={
        "bpm_names": ["B1", "B2"], "horizontal_corrector_names": ["H1"],
        "vertical_corrector_names": ["V1"], "rf_step_hz": 200.0,
        "units": {"orm": "m", "dispersion": "m", "bpm_noise": "m"},
    }, stages=[FitStage("normal", stage_mapping, "original_model")])
    report = preflight_recipe(recipe, lattice_path=str(lattice), measurement_identity={},
                              measurements=paths)
    assert not report["compatible"]
    assert any("selects 2 horizontal correctors" in error for error in report["errors"])
    assert any("fits quads but selects no lattice elements" in error for error in report["errors"])

    comparison_report = preflight_recipe(
        recipe, lattice_path=str(lattice), measurement_identity={},
        measurements=paths, comparison_only=True,
    )
    # Reference-response readiness is independent of FIT stage parameter and
    # magnet selections. Measurement identity/shape checks remain active.
    assert not any("fits quads" in error for error in comparison_report["errors"])
    assert not any("selects 2 horizontal correctors" in error for error in comparison_report["errors"])


def test_preflight_accepts_different_names_only_through_validated_physical_mapping(tmp_path, monkeypatch):
    from pyLOCO.gui.measurement_mapping import find_exact_device_mappings
    import at

    class Element:
            def __init__(self, name, device):
                self.FamName, self.Device = name, device
                self.PolynomA, self.PolynomB = [0.0], [0.0]
                self.Length, self.PassMethod = 1.0, "StrMPoleSymplectic4Pass"

    ring = [Element("BPM_AT_1", "cs/bpm/1"), Element("BPM_AT_2", "cs/bpm/2"),
            Element("HC_AT", "cs/h/1"), Element("VC_AT", "cs/v/1"), Element("Q", "cs/q")]
    lattice = tmp_path / "ring.mat"; lattice.write_bytes(b"mapped ring")
    paths = _modern_measurements(
        tmp_path, bpm_names=("cs/bpm/1", "cs/bpm/2"),
        h_names=("cs/h/1-B1L",), v_names=("cs/v/1-A1L",),
    )
    inventory = {"bpm": ["cs/bpm/1", "cs/bpm/2"],
                 "horizontal_corrector": ["cs/h/1-B1L"],
                 "vertical_corrector": ["cs/v/1-A1L"]}
    device_mapping = find_exact_device_mappings(ring, inventory, lattice_path=lattice)
    stage = mapping("quads")
    stage["MachineElements"] = {
        "bpm_ords": [0, 1], "horizontal_corrector_ords": [2],
        "vertical_corrector_ords": [3], "normal_quadrupole_ords": [4],
    }
    recipe = FitRecipe(data_requirements={
        # Deliberately different historical control-system names.
        "bpm_names": ["OLD1", "OLD2"], "horizontal_corrector_names": ["OLDH"],
        "vertical_corrector_names": ["OLDV"], "rf_step_hz": 200.0,
        "units": {"orm": "m", "dispersion": "m", "bpm_noise": "m"},
    }, stages=[FitStage("mapped", stage, "original_model")])
    monkeypatch.setattr(at, "load_lattice", lambda *_args, **_kwargs: ring)
    report = preflight_recipe(recipe, lattice_path=str(lattice), measurement_identity={},
                              measurements=paths, measurement_device_mapping=device_mapping)
    assert report["compatible"], report["errors"]
    wrong = json.loads(json.dumps(device_mapping)); wrong["mappings"][0]["lattice_ordinal"] = 1
    report = preflight_recipe(recipe, lattice_path=str(lattice), measurement_identity={},
                              measurements=paths, measurement_device_mapping=wrong)
    assert not report["compatible"]
    assert any("mapping is incomplete" in error.lower() for error in report["errors"])


def test_legacy_identity_is_accepted_only_with_recipe_pinned_hashes(tmp_path):
    lattice = tmp_path / "ring.mat"; lattice.write_bytes(b"ring")
    paths = {}
    for role, datasets in (("orm", ("response_matrix",)),
                           ("dispersion", ("measured_eta_x", "measured_eta_y")),
                           ("bpm_noise", ("Noise_BPMx", "Noise_BPMy"))):
        path = tmp_path / f"legacy-{role}.h5"
        with h5py.File(path, "w") as handle:
            for name in datasets:
                shape = (4, 2) if role == "orm" else (2,)
                handle.create_dataset(name, data=np.zeros(shape))
        paths[role] = str(path)
    from pyLOCO.gui.fit_workflow import file_sha256
    requirements = {
        "bpm_names": ["B1", "B2"], "horizontal_corrector_names": ["H1"],
        "vertical_corrector_names": ["V1"], "rf_step_hz": 200.0,
        "units": {"orm": "m", "dispersion": "m", "bpm_noise": "m"},
        "measurement_sha256": {role: file_sha256(path) for role, path in paths.items()},
    }
    recipe = FitRecipe(data_requirements=requirements,
                       stages=[FitStage("one", mapping("quads"), "original_model")])
    report = preflight_recipe(recipe, lattice_path=str(lattice), measurement_identity={},
                              measurements=paths,
                              measurement_options={"dispersion": {"rf_step_hz": 200.0}})
    assert report["compatible"]
    assert "verified by SHA-256" in report["measurement_payload"]["files"]["orm"]["identity_source"]

    with h5py.File(paths["orm"], "r+") as handle:
        handle["response_matrix"][0, 0] = 1.0
    report = preflight_recipe(recipe, lattice_path=str(lattice), measurement_identity={},
                              measurements=paths)
    assert not report["compatible"]
    assert any("recipe-pinned checksum" in error for error in report["errors"])


def test_continuous_and_saved_resume_use_identical_stage2_checkpoint(tmp_path):
    lattice = tmp_path / "ring.mat"; lattice.write_bytes(b"ring")
    base = SimpleNamespace(project_name="test", lattice_path=str(lattice),
                           measurements={"orm": "orm.h5"}, measurement_session={"id": "same"},
                           backend_mapping={"Output": {"directory": str(tmp_path / "results")}})
    recipe = FitRecipe(stages=[FitStage("one", mapping("quads"), "original_model"),
                               FitStage("two", mapping("skew_quads"), "previous_stage")])
    calls = []

    def runner(request, **_):
        root = tmp_path / f"run-{len(calls)}"; root.mkdir()
        for name in ("ring_pyloco.mat", "fit_dict.pkl", "fit_results.npy", "blocks.pkl", "summary.json"):
            (root / name).write_bytes(b"checkpoint")
        calls.append(request.backend_mapping.get("Resume", {}).copy())
        assert request.backend_mapping["Output"]["directory"] == str(tmp_path / "results")
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


def test_multistage_progress_reports_overall_and_current_stage(tmp_path):
    lattice = tmp_path / "ring.mat"; lattice.write_bytes(b"ring")
    base = SimpleNamespace(project_name="test", lattice_path=str(lattice),
                           measurements={"orm": "orm.h5"}, measurement_session={},
                           backend_mapping={"Output": {"directory": str(tmp_path / "results")}})
    recipe = FitRecipe(stages=[FitStage("first", mapping("quads"), "original_model"),
                               FitStage("second", mapping("skew_quads"), "previous_stage")])
    events = []
    calls = []

    def runner(request, progress_callback=None, **_):
        root = tmp_path / f"progress-run-{len(calls)}"; root.mkdir()
        for name in ("ring_pyloco.mat", "fit_dict.pkl", "fit_results.npy", "blocks.pkl", "summary.json"):
            (root / name).write_bytes(b"checkpoint")
        progress_callback({"workflow_fraction": .5, "iteration": 1, "total_iterations": 2,
                           "message": "Halfway through this stage."})
        calls.append(request)
        return SimpleNamespace(results_dir=str(root))

    execute_workflow(base, recipe, session_path=tmp_path / "progress-session.json",
                     runner=runner, progress_callback=events.append)

    halfway = [event for event in events if event.get("message") == "Halfway through this stage."]
    assert [(event["workflow_stage"], event["stage_fraction"], event["workflow_fraction"])
            for event in halfway] == [(1, .5, .25), (2, .5, .75)]


def test_multistage_results_record_the_original_workflow_reference(tmp_path, monkeypatch):
    lattice = tmp_path / "ring.mat"; lattice.write_bytes(b"ring")
    base = SimpleNamespace(project_name="test", lattice_path=str(lattice),
                           measurements={}, measurement_session={},
                           backend_mapping={"Output": {"directory": str(tmp_path / "results")}})
    recipe = FitRecipe(stages=[FitStage("first", mapping("quads"), "original_model"),
                               FitStage("second", mapping("skew_quads"), "previous_stage")])
    observed = []
    rebased = []

    def runner(request, **_):
        root = tmp_path / f"reference-run-{len(observed)}"; root.mkdir()
        for name in ("ring_pyloco.mat", "fit_dict.pkl", "fit_results.npy", "blocks.pkl", "summary.json"):
            (root / name).write_bytes(b"checkpoint")
        observed.append(request.backend_mapping["Workflow"])
        return SimpleNamespace(results_dir=str(root))

    monkeypatch.setattr("pyLOCO.gui.fit_workflow._rebase_workflow_optics",
                        lambda result_dir, reference: rebased.append((result_dir, reference)) or True)
    execute_workflow(base, recipe, session_path=tmp_path / "session.json", runner=runner)

    assert [item["stage"] for item in observed] == [1, 2]
    assert all(item["reference_lattice"] == str(lattice) for item in observed)
    assert [reference for _, reference in rebased] == [str(lattice), str(lattice)]
