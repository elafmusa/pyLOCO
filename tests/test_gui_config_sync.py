from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from pyLOCO.gui.models.project import LocoConfiguration, ProjectMetadata, load_example_project_data
from pyLOCO.gui.models.project import ResponseMatrixConfig
from pyLOCO.gui.backend import (
    _apply_bad_bpm_positions, _apply_machine_element_selections,
    _build_pyloco_kwargs, _make_gui_config,
)
from pyLOCO.gui.main_window import MainWindow


REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("source", "canonical"), [
    ("Linear", "Linear"), ("Analytical", "Analytical"),
    ("Numerical", "Numerical"), ("Tracking", "Numerical"),
])
def test_response_matrix_calculator_project_compatibility(source, canonical):
    assert ResponseMatrixConfig(calculator=source).to_rm_config_kwargs()["calculator"] == canonical


def test_current_yaml_round_trip_preserves_scientific_and_unknown_fields(tmp_path):
    source = REPOSITORY / "Examples" / "measured_machine" / "configs" / "template.yaml"
    config, _, _ = load_example_project_data(source)
    config.source_config["future_backend"] = {"new_option": 17, "nested": {"keep": True}}
    target = tmp_path / "roundtrip.yaml"
    config.save(target)
    reloaded = LocoConfiguration.load(target)

    assert reloaded.solver == config.solver
    assert reloaded.svd == config.svd
    assert reloaded.parameters.fit_list() == config.parameters.fit_list()
    assert reloaded.source_config["future_backend"] == config.source_config["future_backend"]


@pytest.mark.parametrize("relative", [False, True])
def test_constraint_schema_round_trip(relative, tmp_path):
    config = LocoConfiguration(source_config={"loco": {}, "fit_parameters": {}})
    config.constraints.enable = True
    config.constraints.quad_sigma_mode = "relative" if relative else "absolute"
    config.constraints.quad_sigma = 0.02
    config.constraints.quad_relative_sigma = 2e-4
    config.constraints.quad_minimum_sigma = 3e-12
    config.constraints.quad_default_weight = 0.0
    config.constraints.quad_selected_weight = 5.0
    config.constraints.quad_selected_families = [2, 7]
    config.constraints.quad_weighted_families = {9: 3.5}
    target = tmp_path / "constraints.yaml"
    config.save(target)
    reloaded = LocoConfiguration.load(target)

    assert reloaded.constraints.enable
    assert reloaded.constraints.quad_sigma_mode == config.constraints.quad_sigma_mode
    assert reloaded.constraints.quad_selected_weight == 5.0
    assert reloaded.constraints.quad_selected_families == [2, 7]
    assert reloaded.constraints.quad_weighted_families == {9: 3.5}


def test_resume_configuration_and_metadata(tmp_path):
    results = tmp_path / "previous" / "results"
    results.mkdir(parents=True)
    for name in ("ring_pyloco.mat", "fit_dict.pkl", "fit_results.npy"):
        (results / name).touch()
    (results / "summary.json").write_text(json.dumps({
        "chi2_history": [12.0, 4.0], "fit_list": ["quads"], "timestamp": "2026-08-13"
    }))
    config = LocoConfiguration()
    config.resume.enabled = True
    config.resume.directory = str(results.parent)

    assert config.resume.validation_messages() == []
    assert config.resume.metadata()["previous_iterations"] == 2
    assert config.resume.metadata()["previous_final_chi2"] == 4.0
    assert config.to_backend_mapping()["Resume"]["enabled"] is True


def test_basic_advanced_mode_does_not_change_configuration():
    project = ProjectMetadata(mode="Basic")
    project.loco_config.solver.nLMIter = 27
    project.loco_config.parameters.quads_tilt_method = "set"
    before = project.loco_config.to_backend_mapping()
    project.mode = "Advanced"
    project.mode = "Basic"
    assert project.loco_config.to_backend_mapping() == before


@pytest.mark.parametrize("relative_path", [
    "Examples/PETRAIII/measurments/pyloco_config.yaml",
    "Examples/PETRAIII/pyloco_config.yaml",
    "Examples/measured_machine/configs/ebs.yaml",
    "Examples/measured_machine/configs/petra_iii.yaml",
    "Examples/reconstruct_quadrupoles_errors_examples/reconstruct_one_quad_errors/pyloco_config.yaml",
])
def test_representative_repository_configuration_loads(relative_path):
    config, measurements, lattice = load_example_project_data(REPOSITORY / relative_path)
    assert lattice
    if "reconstruct_quadrupoles_errors_examples" not in relative_path:
        assert "orm" in measurements
    assert config.parameters.fit_list()
    assert config.solver.algorithm in {"lm", "gn"}


@pytest.mark.parametrize("relative_path", [
    "Examples/PETRAIII/GUI/petra_iii.pyloco.json",
])
def test_portable_gui_example_is_complete_and_runnable(relative_path):
    project = ProjectMetadata.load(REPOSITORY / relative_path)

    assert project.validation_messages() == []
    assert Path(project.lattice.path).is_file()
    assert Path(project.measurements["orm"].path).is_file()
    cmstep = project.loco_config.parameters.cmstep
    if cmstep.mode == "file":
        assert Path(cmstep.file).is_file()
    assert Path(project.loco_config.output_directory).is_absolute()


def test_petra_skew_attribute_reaches_typed_fit_configuration():
    config, _, _ = load_example_project_data(
        REPOSITORY / "Examples/PETRAIII/pyloco_config_coupling.yaml"
    )
    assert config.source_config["loco"]["skew_attribute"] == "PolynomB"
    assert config.parameters.skew_attr == "PolynomB"
    assert config.parameters.skew_attr_index == 1
    assert config.to_backend_mapping()["FitInitConfig"]["skew_attr"] == "PolynomB"


def test_legacy_gui_project_missing_new_fields_loads():
    legacy = {
        "solver": {"nIter": 3},
        "parameters": {"quads": True, "hbpm_gain": False},
        "constraints": {"enable": False},
    }
    config = LocoConfiguration.from_dict(legacy)
    assert config.solver.nIter == 3
    assert not config.resume.enabled
    assert config.parameters.quads
    assert config.rejection.quad_jacobian_calculator == "Numerical"
    assert config.rejection.skew_jacobian_calculator == "Numerical"


def test_normal_and_skew_jacobian_calculators_are_independent():
    config = LocoConfiguration()
    for normal, skew in (
        ("Numerical", "Numerical"),
        ("Analytical", "Numerical"),
        ("Numerical", "Analytical"),
        ("Analytical", "Analytical"),
    ):
        config.rejection.quad_jacobian_calculator = normal
        config.rejection.skew_jacobian_calculator = skew
        options = config.to_backend_mapping()["LOCOOptions"]
        assert options["quad_jacobian_calculator"] == normal
        assert options["skew_jacobian_calculator"] == skew


@pytest.mark.parametrize(("normal", "skew"), [
    ("Numerical", "Numerical"),
    ("Analytical", "Numerical"),
    ("Numerical", "Analytical"),
    ("Analytical", "Analytical"),
])
def test_jacobian_calculators_round_trip_and_reach_backend_kwargs(
    tmp_path, normal, skew
):
    config = LocoConfiguration(source_config={"loco": {}})
    config.rejection.quad_jacobian_calculator = normal
    config.rejection.skew_jacobian_calculator = skew
    config.rejection.analytical_thick_quadrupole = False
    config.rejection.analytical_thick_steerers = True
    config.rejection.analytical_verbose = True
    config.rejection.analytical_use_mp = True
    config.rejection.analytical_thick_skew = False
    config.rejection.analytical_skew_thick_steerers = True
    config.rejection.analytical_skew_verbose = True
    config.rejection.analytical_skew_use_mp = True
    target = tmp_path / "jacobian-options.json"
    config.save(target)
    reloaded = LocoConfiguration.load(target)
    options = reloaded.to_backend_mapping()["LOCOOptions"]

    indices = {
        "used_bpms_ords": np.arange(2),
        "used_cor_ords": [np.arange(1), np.arange(1)],
        "quads_ords": np.arange(2),
        "skew_ords": np.arange(2),
        "CAVords": np.array([], dtype=int),
        "quads_tilt_ind": np.array([], dtype=int),
        "nHBPM": 2,
        "nVBPM": 2,
        "nHorCOR": 1,
        "nVerCOR": 1,
    }
    kwargs = _build_pyloco_kwargs(
        ring=[],
        options=options,
        rm_cfg=SimpleNamespace(
            dkick=[[1e-5], [1e-5]], rfStep=-3000.0,
            includeDispersion=False, fixedpathlength=False, calculator="Linear",
        ),
        fit_cfg=SimpleNamespace(individuals=True),
        constraint_cfg=None,
        fixed_parameters=SimpleNamespace(
            Frequency=499664399.4230182, rfstep=-3000.0
        ),
        measured={
            "orm": np.zeros((4, 2)),
            "noise_x": np.ones(2),
            "noise_y": np.ones(2),
            "eta_x": np.zeros(2),
            "eta_y": np.zeros(2),
            "dispersion_supplied": True,
        },
        indices=indices,
    )
    assert kwargs["quad_jacobian_calculator"] == normal
    assert kwargs["skew_jacobian_calculator"] == skew
    for key in (
        "analytical_thick_steerers", "analytical_verbose",
        "analytical_use_mp", "analytical_skew_thick_steerers",
        "analytical_skew_verbose", "analytical_skew_use_mp",
    ):
        assert kwargs[key] is True
    assert kwargs["analytical_thick_quadrupole"] is False
    assert kwargs["analytical_thick_skew"] is False


def test_jacobian_gui_controls_restore_and_disable_without_losing_values():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        config = window.project.loco_config
        config.rejection.quad_jacobian_calculator = "Analytical"
        config.rejection.skew_jacobian_calculator = "Analytical"
        config.rejection.analytical_verbose = True
        config.rejection.analytical_skew_verbose = True
        window._load_config_to_widgets()
        assert [window.quad_jacobian_calculator.itemText(index) for index in range(2)] == [
            "Numerical", "Analytical"
        ]
        assert window.quad_jacobian_calculator.currentText() == "Analytical"
        assert window.skew_jacobian_calculator.currentText() == "Analytical"
        assert window.normal_analytical_options.isEnabled()
        assert window.skew_analytical_options.isEnabled()

        window.quad_jacobian_calculator.setCurrentText("Numerical")
        window.skew_jacobian_calculator.setCurrentText("Numerical")
        app.processEvents()
        assert not window.normal_analytical_options.isEnabled()
        assert not window.skew_analytical_options.isEnabled()
        assert window.analytical_verbose.isChecked()
        assert window.analytical_skew_verbose.isChecked()
    finally:
        window.deleteLater()


def test_new_pyloco_options_do_not_break_legacy_options_constructor():
    config = LocoConfiguration()
    config.rejection.skew_individuals = False
    config.rejection.tilt_individuals = False
    config.rejection.calculate_delta_chi2 = True
    mapping = config.to_backend_mapping()

    module = _make_gui_config(mapping)

    assert module.loco_options.individuals is True
    assert mapping["LOCOOptions"]["skew_individuals"] is False
    assert mapping["LOCOOptions"]["tilt_individuals"] is False
    assert mapping["LOCOOptions"]["calculate_delta_chi2"] is True


def test_explicit_machine_selection_is_reduced_with_bad_bpms():
    import numpy as np
    from pyLOCO.pyloco import remove_bad_bpms

    n_bpms, n_hcor, n_vcor = 246, 219, 194
    indices = {
        "used_bpms_ords": np.arange(n_bpms),
        "used_cor_ords": [np.arange(n_hcor), np.arange(n_vcor)],
        "quads_ords": np.arange(398), "skew_ords": np.arange(16),
        "CAVords": np.array([], dtype=int), "quads_tilt_ind": np.arange(398),
        "nHBPM": n_bpms, "nVBPM": n_bpms,
        "nHorCOR": n_hcor, "nVerCOR": n_vcor,
    }
    selections = {
        "bpm_ords": list(range(n_bpms)),
        "horizontal_corrector_ords": list(range(n_hcor)),
        "vertical_corrector_ords": list(range(n_vcor)),
    }
    rm_cfg = SimpleNamespace(bpm_ords=[], cm_ords=None, cav_ords=[])
    selected = _apply_machine_element_selections(indices, selections, rm_cfg)
    measured = {
        "orm": np.zeros((2 * n_bpms, n_hcor + n_vcor)),
        "noise_x": np.ones(n_bpms), "noise_y": np.ones(n_bpms),
        "eta_x": np.zeros(n_bpms), "eta_y": np.zeros(n_bpms),
    }
    bad = np.array([24, 104, 108, 111, 123, 138, 144, 153, 161, 162, 243])

    cleaned, reduced = _apply_bad_bpm_positions(measured, selected, bad, remove_bad_bpms)

    assert cleaned["orm"].shape == (470, 413)
    assert len(reduced["used_bpms_ords"]) == 235
    assert reduced["nHBPM"] == reduced["nVBPM"] == 235
