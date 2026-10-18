#!/usr/bin/env python3
"""Build the local-only modern EBS MDT smoke-test bundle without running FIT.

The command copies authoritative input files unchanged, creates checksum-bound
device mappings, and writes a validated FIT project.  It never connects to a
control system and never invokes the optimizer.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import at

from pyLOCO.gui.fit_workflow import FitRecipe, FitStage, file_sha256, inspect_measurement_files, preflight_recipe
from pyLOCO.gui.lattice_loader import load_project_lattice
from pyLOCO.gui.measurement_mapping import (
    find_exact_device_mappings,
    mapping_counts,
    measurement_inventory,
    ordered_ordinals,
    validate_mapping,
)
from pyLOCO.gui.models.project import ImportedDataset, LatticeSelection, LocoConfiguration, ProjectMetadata, json_safe
from pyLOCO.gui.suite import inspect_measurement_session


QUADRUPLES = ((27, "QD3J"), (36, "QF4J"), (51, "QD5B"))
SELECTION_PATTERNS = {
    "QD3": "QD3[AE]*",
    "QF4": "QF4[ABDE]*",
    "QD5": "QD5[BD]*",
    "skew": "S[HFDIJ]*",
}


def copy_exact(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or file_sha256(source) != file_sha256(target):
        shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--lattice", type=Path, required=True)
    parser.add_argument("--pyaml-config", type=Path, required=True)
    parser.add_argument("--correct-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("mdt_smoke"))
    args = parser.parse_args()

    root = args.output.resolve()
    for child in ("configs", "lattice", "measurements", "mappings", "results"):
        (root / child).mkdir(parents=True, exist_ok=True)

    source_session = inspect_measurement_session(args.session)
    lattice_target = root / "lattice" / "betamodel.mat"
    pyaml_target = root / "configs" / "betamodel_conf_pyaml_test.local.yaml"
    copy_exact(args.lattice.resolve(), lattice_target)
    copy_exact(args.pyaml_config.resolve(), pyaml_target)
    for source in source_session.files.values():
        copy_exact(source, root / "measurements" / source.name)
    copy_exact(args.session.resolve(), root / "measurements" / args.session.name)

    session = inspect_measurement_session(root / "measurements" / args.session.name)
    ring = load_project_lattice(lattice_target, "betamodel")
    if len(ring) != 4434:
        raise ValueError(f"Expected modern 4434-element EBS lattice, got {len(ring)}")
    for ordinal, expected_name in QUADRUPLES:
        actual = str(getattr(ring[ordinal], "FamName", ""))
        if actual != expected_name:
            raise ValueError(f"Quadrupole {ordinal} is {actual!r}, expected {expected_name!r}")

    payload = inspect_measurement_files(
        {role: str(path) for role, path in session.files.items()}, session.options
    )
    if payload["errors"]:
        raise ValueError("; ".join(payload["errors"]))
    inventory = measurement_inventory(payload)
    measurement_mapping = find_exact_device_mappings(
        ring,
        inventory,
        lattice_path=lattice_target,
        configuration_path=pyaml_target,
        provenance="exact AT Device identity from the 18 October 2026 Measure session",
    )
    measurement_mapping = validate_mapping(
        measurement_mapping,
        ring,
        inventory,
        lattice_path=lattice_target,
        configuration_path=pyaml_target,
        require_provenance=True,
    )
    counts = mapping_counts(measurement_mapping)
    if counts["mapped"] != 330 or counts["total"] != 330:
        raise ValueError(f"Expected 330/330 mapped devices, got {counts}")
    mapping_path = root / "mappings" / "EBS_measurement_devices.validated.local.json"
    mapping_path.write_text(json.dumps(measurement_mapping, indent=2), encoding="utf-8")

    bpm_ords = ordered_ordinals(measurement_mapping, "bpm")
    hcor_ords = ordered_ordinals(measurement_mapping, "horizontal_corrector")
    vcor_ords = ordered_ordinals(measurement_mapping, "vertical_corrector")
    assert bpm_ords is not None and hcor_ords is not None and vcor_ords is not None

    config = LocoConfiguration()
    elements = config.machine_elements
    elements.bpm_ords = bpm_ords
    elements.horizontal_corrector_ords = hcor_ords
    elements.vertical_corrector_ords = vcor_ords
    elements.normal_quadrupole_ords = [item[0] for item in QUADRUPLES]
    elements.skew_quadrupole_ords = []
    elements.quadrupole_tilt_ords = []
    elements.cavity_ords = [i for i, element in enumerate(ring) if isinstance(element, at.RFCavity)]

    config.response_matrix.dkick_h = 1e-4
    config.response_matrix.dkick_v = 1e-4
    config.response_matrix.bidirectional = True
    config.response_matrix.includeDispersion = True
    config.response_matrix.rfStep = -200.0
    config.response_matrix.calculator = "Linear"
    config.response_matrix.coupling_orm = False
    config.solver.algorithm = "lm"
    config.solver.nIter = 2
    config.solver.nLMIter = 4
    config.rejection.includeDispersion = True
    config.rejection.hor_dispersion_weight = 1.0
    config.rejection.ver_dispersion_weight = 1.0
    config.rejection.remove_coupling_ = True
    config.constraints.enable = False
    for name in (
        "skew_quads", "quads_tilt", "hbpm_gain", "vbpm_gain", "hbpm_coupling",
        "vbpm_coupling", "hcor_cal", "vcor_cal", "hcor_coupling", "vcor_coupling",
        "HCMEnergyShift", "VCMEnergyShift", "delta_rf",
    ):
        setattr(config.parameters, name, False)
    config.parameters.quads = True
    config.parameters.individuals = True
    config.parameters.cmstep.horizontal = 1e-4
    config.parameters.cmstep.vertical = 1e-4
    config.parameters.rfStep = -200.0
    config.fixed_parameters.Frequency = str(float(ring.rf_frequency))
    config.fixed_parameters.HarmNumber = int(ring.harmonic_number)
    config.fixed_parameters.rfstep = -200.0
    config.output_directory = str(root / "results")
    config.run_name = "EBS_pyaml_MDT_smoke"
    config.element_selection_state = {
        "measurement_mapping": {
            "provenance": "validated exact Device mapping in HDF5 order",
            "inventory_sha256": measurement_mapping["inventory"]["sha256"],
        },
        "normal_quadrupoles": {
            "provenance": "validated pyLOCO Correct machine mapping",
            "ords": [item[0] for item in QUADRUPLES],
            "names": [item[1] for item in QUADRUPLES],
        },
    }
    config._sync_response_matrix_elements()

    requirements = {
        "bpm_names": inventory["bpm"],
        "horizontal_corrector_names": inventory["horizontal_corrector"],
        "vertical_corrector_names": inventory["vertical_corrector"],
        "measurement_sha256": {role: file_sha256(path) for role, path in session.files.items()},
        "units": {role: payload["files"][role].get("unit") for role in session.files},
        "rf_step_hz": float(payload["files"]["dispersion"]["rf_step_hz"]),
        "orm_convention": "unscaled orbit difference [m]; H/V BPM rows and H/V corrector columns",
        "dispersion_convention": payload["files"]["dispersion"].get("measured_eta_definition"),
    }
    snapshot = {
        "gui_config": json_safe(asdict(config)),
        "backend_mapping": json_safe(config.to_backend_mapping()),
    }
    recipe = FitRecipe(
        name="Modern EBS pyAML MDT restricted smoke",
        reference_model_checksum=file_sha256(lattice_target),
        data_requirements=requirements,
        stages=[FitStage("Restricted normal-quadrupole smoke", snapshot, "original_model")],
    )
    project = ProjectMetadata(
        name="Modern EBS pyAML MDT smoke — restricted 3-quadrupole FIT",
        description=(
            "Local-only MDT smoke project using the 18 October 2026 Measure session. "
            "Fits only QD3J, QF4J and QD5B; Run LOCO executes the complete one-stage recipe."
        ),
        mode="Advanced",
        lattice=LatticeSelection(str(lattice_target), "mat", len(ring), "betamodel"),
        measurements={
            role: ImportedDataset(role, str(path), "h5", path.stat().st_size, session.options[role])
            for role, path in session.files.items()
        },
        loco_config=config,
        measurement_session=session.provenance,
        fit_recipe=recipe.to_dict(),
        compatibility_validation={"required": True, "status": "validated", "issues": []},
        measurement_device_mapping=measurement_mapping,
    )
    # Treat the whole mdt_smoke directory as the portable project bundle so
    # ProjectMetadata.save() writes sibling lattice/measurement/result paths
    # relative to configs/ instead of baking in this workstation's checkout.
    project.base_directory = str(root / "configs")
    project_path = project.save(root / "configs" / "EBS_pyaml_MDT_smoke.pyloco.json")

    # Resolve the historical scientific selection strategy on this exact
    # modern lattice. Pattern matches are retained as auditable provenance;
    # no historical ordinal is copied.
    selections: dict[str, list[int]] = {
        label: [int(value) for value in at.get_refpts(ring, pattern)]
        for label, pattern in SELECTION_PATTERNS.items()
    }
    normal_ords = sorted(selections["QD3"] + selections["QF4"] + selections["QD5"])
    skew_ords = selections["skew"]
    if len(normal_ords) != len(set(normal_ords)):
        raise ValueError("Normal-quadrupole selection patterns overlap")
    if len(skew_ords) != len(set(skew_ords)):
        raise ValueError("Skew selection pattern produced duplicate ordinals")
    skew_invalid = [
        ordinal for ordinal in skew_ords
        if not hasattr(ring[ordinal], "PolynomA") or len(ring[ordinal].PolynomA) <= 1
    ]
    if skew_invalid:
        raise ValueError(f"Skew candidates lack PolynomA[1]: {skew_invalid}")
    selection_report = {
        "reference_lattice": {
            "path": "../lattice/betamodel.mat",
            "sha256": file_sha256(lattice_target),
            "element_count": len(ring),
            "mat_variable": "betamodel",
        },
        "strategy": {
            "normal": [SELECTION_PATTERNS[key] for key in ("QD3", "QF4", "QD5")],
            "skew": SELECTION_PATTERNS["skew"],
            "ordinal_convention": "zero-based AT lattice ordinal",
        },
        "groups": {},
        "unmatched_patterns": [],
        "ambiguous_elements": [],
    }
    for label, ordinals in selections.items():
        selection_report["groups"][label] = {
            "pattern": SELECTION_PATTERNS[label],
            "count": len(ordinals),
            "elements": [
                {
                    "ordinal": ordinal,
                    "name": str(ring[ordinal].FamName),
                    "physical_type": type(ring[ordinal]).__name__,
                }
                for ordinal in ordinals
            ],
        }
    selection_report["normal_total"] = len(normal_ords)
    selection_report["skew_total"] = len(skew_ords)
    selection_report["skew_polynom_a1_validated"] = len(skew_ords) - len(skew_invalid)
    selection_path = root / "configs" / "EBS_modern_selection_provenance.json"
    selection_path.write_text(json.dumps(selection_report, indent=2), encoding="utf-8")

    def stage_config(fit_groups: list[str], *, remove_coupling: bool,
                     dispersion_weight: float, svd_cut: int) -> LocoConfiguration:
        stage = deepcopy(config)
        stage.machine_elements.normal_quadrupole_ords = list(normal_ords)
        stage.machine_elements.skew_quadrupole_ords = list(skew_ords)
        stage.machine_elements.quadrupole_tilt_ords = []
        stage.element_selection_state["normal_quadrupoles"] = {
            "provenance": "patterns resolved on checksum-bound modern lattice",
            "patterns": [SELECTION_PATTERNS[key] for key in ("QD3", "QF4", "QD5")],
            "selection_report": "EBS_modern_selection_provenance.json",
        }
        stage.element_selection_state["skew_quadrupoles"] = {
            "provenance": "pattern resolved and PolynomA[1] checked on checksum-bound modern lattice",
            "pattern": SELECTION_PATTERNS["skew"],
            "selection_report": "EBS_modern_selection_provenance.json",
        }
        for name in (
            "quads", "skew_quads", "quads_tilt", "hbpm_gain", "vbpm_gain",
            "hbpm_coupling", "vbpm_coupling", "hcor_cal", "vcor_cal",
            "hcor_coupling", "vcor_coupling", "HCMEnergyShift", "VCMEnergyShift", "delta_rf",
        ):
            setattr(stage.parameters, name, name in fit_groups)
        stage.solver.nIter = 2
        stage.solver.nLMIter = 10
        stage.svd.svd_selection_method = "user_input"
        stage.svd.cut_ = svd_cut
        stage.rejection.remove_coupling_ = remove_coupling
        stage.rejection.hor_dispersion_weight = dispersion_weight
        stage.rejection.ver_dispersion_weight = dispersion_weight
        # Historical positional exclusions are deliberately not carried into
        # a new identity inventory without an explicit physical revalidation.
        stage.bad_bpm_positions = []
        stage.excluded_horizontal_corrector_positions = []
        stage.excluded_vertical_corrector_positions = []
        stage._sync_response_matrix_elements()
        return stage

    stage_specs = (
        ("Calibration", ["hbpm_gain", "vbpm_gain", "hcor_cal", "vcor_cal"], True, 5.0, 100),
        ("BPM coupling", ["hbpm_coupling", "vbpm_coupling"], False, 10.0, 0),
        ("Normal optics", ["quads"], True, 5.0, 110),
        ("Skew and coupling", ["skew_quads", "hbpm_coupling", "vbpm_coupling"], False, 5.0, 50),
    )
    extended_stages = []
    extended_configs = []
    for position, (name, groups, remove_coupling, weight, cut) in enumerate(stage_specs):
        stage = stage_config(groups, remove_coupling=remove_coupling,
                             dispersion_weight=weight, svd_cut=cut)
        extended_configs.append(stage)
        extended_stages.append(FitStage(
            name,
            {"gui_config": json_safe(asdict(stage)),
             "backend_mapping": json_safe(stage.to_backend_mapping())},
            "original_model" if position == 0 else "previous_stage",
        ))
    extended_recipe = FitRecipe(
        name="Modern EBS pyAML MDT four-stage proposal — operator review required",
        reference_model_checksum=file_sha256(lattice_target),
        data_requirements=deepcopy(requirements),
        stages=extended_stages,
    )
    extended_project = ProjectMetadata(
        name="Modern EBS pyAML MDT four-stage proposal — REVIEW REQUIRED",
        description=(
            "Historical Case C selection strategy re-resolved on the modern 4434-element lattice. "
            "The current 5H/5V ORM is retained unchanged but is not claimed to constrain this extended fit."
        ),
        mode="Advanced",
        lattice=LatticeSelection(str(lattice_target), "mat", len(ring), "betamodel"),
        measurements=deepcopy(project.measurements),
        loco_config=extended_configs[0],
        measurement_session=deepcopy(session.provenance),
        fit_recipe=extended_recipe.to_dict(),
        compatibility_validation={
            "required": True,
            "status": "review_required",
            "issues": [
                "Operator/scientist review required before execution.",
                "The current ORM has only 5 H and 5 V correctors and is not claimed to sufficiently constrain the extended fit.",
                "Historical BPM exclusion positions [27, 231, 286] were not applied because their physical identities were not revalidated against this session.",
            ],
        },
        measurement_device_mapping=deepcopy(measurement_mapping),
    )
    extended_project.base_directory = str(root / "configs")
    extended_path = extended_project.save(
        root / "configs" / "EBS_pyaml_MDT_four_stage_REVIEW_REQUIRED.pyloco.json"
    )

    correct_mapping = json.loads(args.correct_mapping.read_text(encoding="utf-8"))
    if correct_mapping.get("schema") != "pyloco.machine_mapping" or correct_mapping.get("schema_version") != 2:
        raise ValueError("Correct mapping is not pyloco.machine_mapping schema v2")
    if correct_mapping.get("reference_lattice", {}).get("sha256") != file_sha256(lattice_target):
        raise ValueError("Correct mapping lattice checksum differs from the packaged lattice")
    expected = {(ordinal, name) for ordinal, name in QUADRUPLES}
    actual = {(row.get("lattice_ordinal"), row.get("lattice_name")) for row in correct_mapping.get("mappings", [])}
    if actual != expected or any(row.get("component") != "B2L" or row.get("unit") != "m^-1" for row in correct_mapping["mappings"]):
        raise ValueError("Correct mapping does not contain the exact three validated B2L/m^-1 entries")
    correct_mapping["reference_lattice"]["path"] = "../lattice/betamodel.mat"
    correct_target = root / "mappings" / "EBS_three_quadrupoles.correct.validated.json"
    correct_target.write_text(json.dumps(correct_mapping, indent=2), encoding="utf-8")

    report = preflight_recipe(
        recipe,
        lattice_path=str(lattice_target),
        measurement_identity=project.measurement_session,
        lattice_element_names=[str(getattr(element, "FamName", "")) for element in ring],
        measurements={role: str(path) for role, path in session.files.items()},
        measurement_options=session.options,
        measurement_device_mapping=measurement_mapping,
        lattice_mat_variable="betamodel",
    )
    extended_preflight = preflight_recipe(
        extended_recipe,
        lattice_path=str(lattice_target),
        measurement_identity=extended_project.measurement_session,
        lattice_element_names=[str(getattr(element, "FamName", "")) for element in ring],
        measurements={role: str(path) for role, path in session.files.items()},
        measurement_options=session.options,
        measurement_device_mapping=measurement_mapping,
        lattice_mat_variable="betamodel",
    )
    validation = {
        "project": str(project_path),
        "project_validation_messages": project.validation_messages(),
        "recipe_preflight": report,
        "mapping_counts": counts,
        "expected_orm_shape": [640, 10],
        "fit_list": config.parameters.fit_list(),
        "normal_quadrupoles": [{"ordinal": ordinal, "name": name} for ordinal, name in QUADRUPLES],
        "checksums": {
            "lattice": file_sha256(lattice_target),
            "pyaml_configuration": file_sha256(pyaml_target),
            "session_manifest": file_sha256(root / "measurements" / args.session.name),
            **{role: file_sha256(path) for role, path in session.files.items()},
        },
        "optimizer_executed": False,
        "machine_writes": False,
        "extended_four_stage_proposal": {
            "project": str(extended_path),
            "recipe_preflight": extended_preflight,
            "execution_gate": "operator/scientist review required",
            "normal_quadrupole_count": len(normal_ords),
            "skew_candidate_count": len(skew_ords),
            "current_orm_correctors": {"horizontal": len(hcor_ords), "vertical": len(vcor_ords)},
            "historical_bpm_exclusions_applied": False,
        },
    }
    (root / "VALIDATION.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if validation["project_validation_messages"] or report["errors"] or extended_preflight["errors"]:
        raise ValueError(json.dumps(validation, indent=2))
    print(json.dumps(validation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
