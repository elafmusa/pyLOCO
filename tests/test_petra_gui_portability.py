"""Portability checks for the tracked PETRA III GUI example project."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from pyLOCO.gui.models.project import ProjectMetadata
from pyLOCO.gui.backend import LocoRunRequest


REPOSITORY = Path(__file__).resolve().parents[1]
PETRA_ROOT = REPOSITORY / "Examples" / "PETRAIII"
PROJECT_FILE = PETRA_ROOT / "GUI" / "petra_iii.pyloco.json"
COUPLING_PROJECT_FILE = PETRA_ROOT / "GUI" / "petra_iii_coupling.pyloco.json"


def test_petra_gui_project_contains_no_machine_specific_absolute_paths():
    raw = PROJECT_FILE.read_text(encoding="utf-8")
    data = json.loads(raw)

    assert "/Users/" not in raw
    assert "/home/" not in raw
    assert str(REPOSITORY) not in raw

    path_values = (
        data["path"],
        data["lattice"]["path"],
        data["measurements"]["orm"]["path"],
        data["measurements"]["dispersion"]["path"],
        data["measurements"]["bpm_noise"]["path"],
        data["loco_config"]["parameters"]["cmstep"]["file"],
        data["loco_config"]["output_directory"],
        data["loco_config"]["resume"]["directory"],
        data["loco_config"]["source_path"],
    )
    assert all(not Path(value).is_absolute() for value in path_values)
    assert data["recent_projects"] == []


def test_petra_gui_project_loads_and_validates_after_relocation(tmp_path):
    relocated = tmp_path / "different-clone" / "Examples" / "PETRAIII"
    gui_dir = relocated / "GUI"
    data_dir = relocated / "data"
    gui_dir.mkdir(parents=True)
    data_dir.mkdir()

    shutil.copy2(PROJECT_FILE, gui_dir / PROJECT_FILE.name)
    shutil.copy2(PETRA_ROOT / "pyloco_config_coupling.yaml", relocated)
    for name in (
        "p3_low_beta.mat",
        "measured_orm_loco.h5",
        "measured_dispersion_loco.h5",
        "measured_BPM_noise_loco.h5",
        "CMstep.npz",
    ):
        shutil.copy2(PETRA_ROOT / "data" / name, data_dir / name)

    project = ProjectMetadata.load(gui_dir / PROJECT_FILE.name)

    assert project.validation_messages() == []
    assert Path(project.lattice.path) == (data_dir / "p3_low_beta.mat").resolve()
    assert Path(project.measurements["orm"].path) == (
        data_dir / "measured_orm_loco.h5"
    ).resolve()
    assert Path(project.measurements["dispersion"].path) == (
        data_dir / "measured_dispersion_loco.h5"
    ).resolve()
    assert Path(project.measurements["bpm_noise"].path) == (
        data_dir / "measured_BPM_noise_loco.h5"
    ).resolve()
    assert Path(project.loco_config.parameters.cmstep.file) == (
        data_dir / "CMstep.npz"
    ).resolve()
    assert Path(project.loco_config.source_path) == (
        relocated / "pyloco_config_coupling.yaml"
    ).resolve()
    expected_output = (relocated / "output" / "measured_orm").resolve()
    assert Path(project.loco_config.output_directory) == expected_output
    assert Path(project.loco_config.resume.directory) == expected_output
    project.save()
    saved_raw = (gui_dir / PROJECT_FILE.name).read_text(encoding="utf-8")
    assert str(tmp_path) not in saved_raw


def _portable_project_at(root: Path) -> Path:
    gui = root / "Examples" / "PETRAIII" / "GUI"
    data = root / "Examples" / "PETRAIII" / "data"
    gui.mkdir(parents=True)
    data.mkdir(parents=True)
    (data / "ring.mat").write_bytes(b"ring")
    (data / "orm.h5").write_bytes(b"orm")
    project = {
        "name": "portable",
        "lattice": {"path": "../data/ring.mat", "file_type": "mat"},
        "measurements": {
            "orm": {"role": "orm", "path": "../data/orm.h5", "file_type": "h5"}
        },
        "loco_config": {},
    }
    target = gui / "portable.pyloco.json"
    target.write_text(json.dumps(project), encoding="utf-8")
    return target


def test_project_base_directory_and_save_remain_portable_in_two_clones(tmp_path):
    for owner in ("user_A", "user_B"):
        project_path = _portable_project_at(tmp_path / owner / "pyLOCO")
        project = ProjectMetadata.load(project_path)
        assert project.path == str(project_path.resolve())
        assert project.base_directory == str(project_path.parent.resolve())
        assert project.resolve_path(project.lattice.path).exists()
        assert project.resolve_path(project.measurements["orm"].path).exists()

        assert project.save() == project_path.resolve()
        raw = project_path.read_text(encoding="utf-8")
        saved = json.loads(raw)
        assert str(tmp_path / owner) not in raw
        assert "path" not in saved
        assert "recent_projects" not in saved
        assert saved["lattice"]["path"] == "../data/ring.mat"

        reopened = ProjectMetadata.load(project_path)
        assert reopened.resolve_path(reopened.lattice.path).exists()
        assert reopened.resolve_path(reopened.measurements["orm"].path).exists()
        request = LocoRunRequest.from_project(reopened)
        assert Path(request.lattice_path).is_absolute()
        assert Path(request.measurements["orm"]).is_absolute()


def test_save_as_rebases_portable_references_and_updates_project_location(tmp_path):
    original = _portable_project_at(tmp_path / "clone" / "pyLOCO")
    project = ProjectMetadata.load(original)
    lattice = project.resolve_path(project.lattice.path)
    destination = tmp_path / "student-project" / "copy.pyloco.json"
    destination.parent.mkdir()

    project.save(destination)
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert not Path(saved["lattice"]["path"]).is_absolute()
    assert project.path == str(destination.resolve())
    assert project.base_directory == str(destination.parent.resolve())
    reopened = ProjectMetadata.load(destination)
    assert reopened.resolve_path(reopened.lattice.path) == lattice


def test_project_json_safely_serializes_nested_numpy_configuration(tmp_path):
    project = ProjectMetadata(name="numpy-safe")
    project.loco_config.source_config = {
        "RMConfig": {
            "dkick": np.asarray([1.0e-5, 2.0e-5]),
            "count": np.int64(2),
            "scale": np.float64(1.25),
            "enabled": np.bool_(True),
        }
    }
    target = project.save(tmp_path / "numpy.pyloco.json")
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["loco_config"]["source_config"]["RMConfig"] == {
        "dkick": [1.0e-5, 2.0e-5], "count": 2, "scale": 1.25, "enabled": True
    }
    restored = ProjectMetadata.load(target)
    assert restored.loco_config.source_config["RMConfig"]["dkick"] == [1.0e-5, 2.0e-5]


def test_missing_relative_resource_reports_exact_project_relative_path(tmp_path):
    project_file = tmp_path / "GUI" / "missing.pyloco.json"
    project_file.parent.mkdir()
    project_file.write_text(json.dumps({
        "lattice": {"path": "../data/does-not-exist.mat", "file_type": "mat"},
        "loco_config": {},
    }), encoding="utf-8")
    project = ProjectMetadata.load(project_file)
    missing = (tmp_path / "data" / "does-not-exist.mat").resolve()
    assert project.resolve_path("../data/does-not-exist.mat") == missing
    assert any(str(missing) in message for message in project.validation_messages())


def test_validated_petra_coupling_project_mapping_and_round_trip(tmp_path):
    raw = COUPLING_PROJECT_FILE.read_text(encoding="utf-8")
    assert "/Users/" not in raw and "/home/" not in raw
    project = ProjectMetadata.load(COUPLING_PROJECT_FILE)
    assert project.validation_messages() == []
    mapping = project.loco_config.to_backend_mapping()
    elements = mapping["MachineElements"]
    assert {key: len(value) for key, value in elements.items()} == {
        "bpm_ords": 246,
        "horizontal_corrector_ords": 40,
        "vertical_corrector_ords": 40,
        "normal_quadrupole_ords": 399,
        "skew_quadrupole_ords": 16,
        "cavity_ords": 12,
    }
    expected_fit = [
        "hbpm_gain", "hbpm_coupling", "vbpm_coupling", "vbpm_gain",
        "hcor_cal", "vcor_cal", "hcor_coupling", "vcor_coupling",
        "quads", "skew_quads", "quads_tilt",
    ]
    selected = set(mapping["FitInitConfig"]["fit_list"])
    from pyLOCO.config import BLOCK_ORDER
    assert [name for name in BLOCK_ORDER if name in selected] == expected_fit
    assert mapping["FitInitConfig"]["individuals"] is False
    assert mapping["BadBPMPositions"] == [71, 92, 101, 104, 108, 123, 138, 153, 161, 243]
    assert mapping["RMConfig"]["dkick"] == (1.0e-4, 1.0e-4)
    assert mapping["RMConfig"]["rfStep"] == -3000.0
    assert mapping["RMConfig"]["calculator"] == "Linear"
    assert mapping["RMConfig"]["coupling_orm"] is False
    options = mapping["LOCOOptions"]
    assert (options["algorithm"], options["nIter"], options["nLMIter"]) == ("lm", 1, 10)
    assert options["Starting_Lambda"] == 1.0e-3 and options["max_lm_lambda"] == 15.0
    assert options["scaled"] is True
    assert options["svd_selection_method"] == "threshold"
    assert options["svd_threshold"] == 1.0e-6 and options["show_svd_plot"] is False
    assert options["apply_normalization"] is True
    assert options["normalization_mode"] == "component"
    assert options["includeDispersion"] is True
    assert options["hor_dispersion_weight"] == options["ver_dispersion_weight"] == 5.0
    assert options["remove_coupling_"] is False
    assert options["quad_jacobian_calculator"] == "Numerical"
    assert options["skew_jacobian_calculator"] == "Numerical"
    assert mapping["ConstraintConfig"]["enable"] is False

    round_trip = COUPLING_PROJECT_FILE.parent / (
        f".{COUPLING_PROJECT_FILE.stem}-{tmp_path.name}.pyloco.json"
    )
    try:
        project.save(round_trip)
        saved = round_trip.read_text(encoding="utf-8")
        assert "/Users/" not in saved and "/home/" not in saved
        reopened = ProjectMetadata.load(round_trip)
        assert reopened.validation_messages() == []
        assert reopened.loco_config.to_backend_mapping() == mapping
    finally:
        round_trip.unlink(missing_ok=True)
