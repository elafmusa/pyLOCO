"""Portability checks for the tracked PETRA III GUI example project."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pyLOCO.gui.models.project import ProjectMetadata


REPOSITORY = Path(__file__).resolve().parents[1]
PETRA_ROOT = REPOSITORY / "Examples" / "PETRAIII"
PROJECT_FILE = PETRA_ROOT / "GUI" / "petra_iii.pyloco.json"


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
