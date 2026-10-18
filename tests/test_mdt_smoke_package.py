from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest
from PySide6.QtWidgets import QApplication

from pyLOCO.gui.measurement_mapping import mapping_counts
from pyLOCO.gui.models.project import ProjectMetadata
from pyLOCO.gui.main_window import MainWindow


ROOT = Path(__file__).resolve().parents[1] / "mdt_smoke"
PROJECT = ROOT / "configs" / "EBS_pyaml_MDT_smoke.pyloco.json"
EXTENDED = ROOT / "configs" / "EBS_pyaml_MDT_four_stage_REVIEW_REQUIRED.pyloco.json"


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.skipif(not PROJECT.is_file(), reason="local-only MDT artifacts are not installed")
def test_modern_ebs_mdt_project_is_restricted_and_complete():
    project = ProjectMetadata.load(PROJECT)
    assert project.lattice.element_count == 4434
    assert project.lattice.mat_variable == "betamodel"
    assert project.validation_messages() == []
    assert project.loco_config.parameters.fit_list() == ["quads"]
    assert project.loco_config.machine_elements.normal_quadrupole_ords == [27, 36, 51]
    assert project.loco_config.machine_elements.skew_quadrupole_ords == []
    assert project.loco_config.machine_elements.quadrupole_tilt_ords == []
    assert len(project.loco_config.machine_elements.bpm_ords) == 320
    assert len(project.loco_config.machine_elements.horizontal_corrector_ords) == 5
    assert len(project.loco_config.machine_elements.vertical_corrector_ords) == 5
    assert mapping_counts(project.measurement_device_mapping) == {
        "mapped": 330,
        "unresolved": 0,
        "ambiguous": 0,
        "duplicate": 0,
        "invalid": 0,
        "total": 330,
    }
    assert len(project.fit_recipe["stages"]) == 1
    assert project.fit_recipe["stages"][0]["start_from"] == "original_model"


@pytest.mark.skipif(not PROJECT.is_file(), reason="local-only MDT artifacts are not installed")
def test_modern_ebs_mdt_build_validation_is_fail_closed():
    report = json.loads((ROOT / "VALIDATION.json").read_text(encoding="utf-8"))
    assert report["project_validation_messages"] == []
    assert report["recipe_preflight"]["compatible"] is True
    assert report["recipe_preflight"]["errors"] == []
    assert report["expected_orm_shape"] == [640, 10]
    assert report["fit_list"] == ["quads"]
    assert report["optimizer_executed"] is False
    assert report["machine_writes"] is False


@pytest.mark.skipif(not EXTENDED.is_file(), reason="local-only MDT artifacts are not installed")
def test_modern_ebs_four_stage_proposal_is_separate_and_review_gated():
    project = ProjectMetadata.load(EXTENDED)
    stages = project.fit_recipe["stages"]
    assert [stage["name"] for stage in stages] == [
        "Calibration", "BPM coupling", "Normal optics", "Skew and coupling"
    ]
    assert [stage["start_from"] for stage in stages] == [
        "original_model", "previous_stage", "previous_stage", "previous_stage"
    ]
    assert [stage["configuration"]["backend_mapping"]["FitInitConfig"]["fit_list"] for stage in stages] == [
        ["hbpm_gain", "vbpm_gain", "hcor_cal", "vcor_cal"],
        ["hbpm_coupling", "vbpm_coupling"],
        ["quads"],
        ["skew_quads", "hbpm_coupling", "vbpm_coupling"],
    ]
    normal = project.loco_config.machine_elements.normal_quadrupole_ords
    skew = project.loco_config.machine_elements.skew_quadrupole_ords
    assert len(normal) == 252 and len(set(normal)) == 252
    assert len(skew) == 288 and len(set(skew)) == 288
    assert project.loco_config.bad_bpm_positions == []
    assert project.compatibility_validation["status"] == "review_required"
    assert not project.is_complete


def test_modern_ebs_selection_provenance_records_exact_matches():
    report = json.loads(
        (ROOT / "configs" / "EBS_modern_selection_provenance.json").read_text(encoding="utf-8")
    )
    assert report["normal_total"] == 252
    assert report["skew_total"] == 288
    assert report["skew_polynom_a1_validated"] == 288
    assert report["unmatched_patterns"] == []
    assert report["ambiguous_elements"] == []
    assert {key: report["groups"][key]["count"] for key in ("QD3", "QF4", "QD5", "skew")} == {
        "QD3": 62, "QF4": 126, "QD5": 64, "skew": 288
    }


@pytest.mark.skipif(not PROJECT.is_file(), reason="local-only MDT artifacts are not installed")
def test_smoke_project_opens_through_real_gui_path(app):
    window = MainWindow()
    window.open_project(PROJECT)
    app.processEvents()
    assert window.project.name.startswith("Modern EBS pyAML MDT smoke")
    assert Path(window.project.lattice.path).is_file()
    assert window.project.lattice.element_count == 4434
    assert window.project.lattice.mat_variable == "betamodel"
    assert set(window.project.measurements) == {"bpm_noise", "dispersion", "orm"}
    assert all(Path(dataset.path).is_file() for dataset in window.project.measurements.values())
    assert len(window.project.measurement_device_mapping["mappings"]) == 330
    assert window.project.loco_config.machine_elements.normal_quadrupole_ords == [27, 36, 51]
    assert [stage.name for stage in window.fit_recipe.stages] == ["Restricted normal-quadrupole smoke"]
    window.project.modified = False
    window.close()


@pytest.mark.skipif(not PROJECT.is_file(), reason="local-only MDT artifacts are not installed")
def test_smoke_project_save_as_close_reopen_and_relocation(app, tmp_path):
    first = MainWindow()
    first.open_project(PROJECT)
    saved = tmp_path / "saved-as.pyloco.json"
    assert first._save_project_to(saved)
    first.project.modified = False
    first.close()
    reopened = MainWindow()
    reopened.open_project(saved)
    app.processEvents()
    assert reopened.project.lattice.element_count == 4434
    assert len(reopened.project.measurements) == 3
    reopened.project.modified = False
    reopened.close()

    relocated = tmp_path / "relocated"
    for directory in ("configs", "lattice", "measurements", "mappings", "results"):
        shutil.copytree(ROOT / directory, relocated / directory)
    moved = MainWindow()
    moved.open_project(relocated / "configs" / PROJECT.name)
    app.processEvents()
    assert Path(moved.project.lattice.path).is_file()
    assert all(relocated in Path(dataset.path).parents for dataset in moved.project.measurements.values())
    profile = moved.project.measurement_device_mapping["machine_profile"]
    assert relocated in Path(profile["configuration_path"]).parents
    moved.project.modified = False
    moved.close()


def test_harmonic_number_string_is_normalized_for_legacy_projects(tmp_path):
    project = ProjectMetadata(name="legacy harmonic")
    target = project.save(tmp_path / "legacy.pyloco.json")
    data = json.loads(target.read_text(encoding="utf-8"))
    data["loco_config"]["fixed_parameters"]["HarmNumber"] = "992"
    target.write_text(json.dumps(data), encoding="utf-8")
    restored = ProjectMetadata.load(target)
    assert restored.loco_config.fixed_parameters.HarmNumber == 992
    assert isinstance(restored.loco_config.fixed_parameters.HarmNumber, int)
