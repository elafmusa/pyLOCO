#!/usr/bin/env python3
"""Run the 2026 EBS MDT four-stage LOCO workflow without Qt or machine I/O."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from pyLOCO.gui.backend import LocoRunRequest, run_loco_request
from pyLOCO.gui.fit_workflow import (
    FitRecipe,
    FitRunSession,
    execute_workflow,
    file_sha256,
    inspect_measurement_files,
    preflight_recipe,
)
from pyLOCO.gui.lattice_loader import load_project_lattice
from pyLOCO.gui.models.project import ProjectMetadata


HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = HERE / "EBS_MDT_20261024.pyloco.json"
DEFAULT_RECIPE = HERE / "recipe.json"


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--session", type=Path, default=HERE / "fit-run-session.json")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run the same four stages with one outer/LM iteration per stage",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _metadata_json(handle) -> dict[str, Any]:
    if "metadata/json" not in handle:
        return {}
    raw = handle["metadata/json"][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _lattice_rf_identity(ring) -> tuple[float, int]:
    frequencies = {
        float(element.Frequency)
        for element in ring
        if hasattr(element, "Frequency") and np.isfinite(float(element.Frequency))
    }
    harmonics = {
        int(element.HarmNumber)
        for element in ring
        if hasattr(element, "HarmNumber") and int(element.HarmNumber) > 0
    }
    if len(frequencies) != 1:
        raise ValueError(f"Reference lattice has {len(frequencies)} distinct RF frequencies.")
    if len(harmonics) != 1:
        raise ValueError(f"Reference lattice has {len(harmonics)} distinct harmonic numbers.")
    return frequencies.pop(), harmonics.pop()


def _mapping_by_role(project: ProjectMetadata) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in (project.machine_mapping or {}).get("mappings", []):
        grouped.setdefault(str(row.get("role", "")), []).append(row)
    return grouped


def scientific_preflight(project: ProjectMetadata, recipe: FitRecipe) -> dict[str, Any]:
    """Validate the exact on-disk payload used by both CLI and GUI backends."""
    import h5py

    errors = list(project.validation_messages())
    request = LocoRunRequest.from_project(project)
    lattice_path = Path(request.lattice_path)
    if not lattice_path.is_file():
        errors.append(f"Reference lattice is missing: {lattice_path}")
        return {"compatible": False, "errors": errors}

    ring = load_project_lattice(request.lattice_path, request.lattice_mat_variable or None)
    lattice_frequency, lattice_harmonic = _lattice_rf_identity(ring)
    if project.lattice.element_count and len(ring) != project.lattice.element_count:
        errors.append(
            f"Reference lattice contains {len(ring)} elements; project records "
            f"{project.lattice.element_count}."
        )

    report = preflight_recipe(
        recipe,
        lattice_path=request.lattice_path,
        lattice_mat_variable=request.lattice_mat_variable or None,
        measurement_identity=request.measurement_session,
        measurements=request.measurements,
        measurement_options=request.measurement_options,
        measurement_device_mapping=project.measurement_device_mapping,
    )
    errors.extend(report["errors"])
    payload = report.get("measurement_payload") or inspect_measurement_files(
        request.measurements, request.measurement_options
    )

    orm = payload.get("files", {}).get("orm", {})
    bpms = list(orm.get("bpm_names", []))
    horizontal = list(orm.get("horizontal_corrector_names", []))
    vertical = list(orm.get("vertical_corrector_names", []))
    if orm.get("shape") != [2 * len(bpms), len(horizontal) + len(vertical)]:
        errors.append("ORM dimensions do not match its ordered BPM/H/V identities.")
    if (len(bpms), len(horizontal), len(vertical)) != (320, 5, 5):
        errors.append(
            "This MDT example requires the verified 320 BPM + 5 H + 5 V inventory; "
            f"found {len(bpms)} + {len(horizontal)} + {len(vertical)}."
        )
    if orm.get("unit") not in {"m", "m/rad"}:
        errors.append(f"Unsupported ORM unit: {orm.get('unit')!r}.")

    with h5py.File(request.measurements["orm"], "r") as handle:
        for plane, count in (("horizontal", len(horizontal)), ("vertical", len(vertical))):
            key = f"kicks/{plane}/actual"
            if key not in handle:
                errors.append(f"ORM is missing authoritative applied kicks: {key}.")
                continue
            kicks = np.asarray(handle[key], dtype=float)
            if kicks.shape != (count,) or not np.isfinite(kicks).all() or np.any(kicks == 0.0):
                errors.append(f"ORM {plane} kick vector is invalid: shape {kicks.shape}.")

    dispersion = payload.get("files", {}).get("dispersion", {})
    if dispersion.get("rf_step_hz") is None or dispersion.get("rf_step_hz", 0.0) <= 0:
        errors.append("Dispersion did not normalize to a positive total RF separation.")
    with h5py.File(request.measurements["dispersion"], "r") as handle:
        metadata = _metadata_json(handle)
        measured_frequency = float(
            metadata.get("nominal_rf_hz", handle.attrs.get("rf_original_hz", math.nan))
        )
    tolerance_hz = max(1e-6, abs(lattice_frequency) * 1e-12)
    if not np.isfinite(measured_frequency) or not math.isclose(
        measured_frequency, lattice_frequency, rel_tol=0.0, abs_tol=tolerance_hz
    ):
        errors.append(
            f"Measured RF frequency {measured_frequency!r} Hz does not match the "
            f"reference lattice {lattice_frequency:.12g} Hz."
        )

    grouped = _mapping_by_role(project)
    for role, expected in (("bpm", bpms), ("horizontal_corrector", horizontal),
                           ("vertical_corrector", vertical)):
        lookup = {str(row.get("pyaml_device")): row for row in grouped.get(role, [])}
        missing = [name for name in expected if name not in lookup]
        if missing:
            errors.append(f"Machine Mapping lacks {role} identity {missing[0]!r}.")

    for index, stage in enumerate(recipe.stages, 1):
        mapping = stage.configuration.get("backend_mapping", stage.configuration)
        fixed = mapping.get("FixedParameters", {})
        configured_frequency = float(fixed.get("Frequency", math.nan))
        configured_harmonic = int(fixed.get("HarmNumber", 0))
        if not math.isclose(configured_frequency, lattice_frequency, rel_tol=0.0,
                            abs_tol=tolerance_hz):
            errors.append(
                f"Stage {index} RF frequency {configured_frequency!r} Hz does not match "
                f"the lattice value {lattice_frequency:.12g} Hz."
            )
        if configured_harmonic != lattice_harmonic:
            errors.append(
                f"Stage {index} harmonic number {configured_harmonic} does not match "
                f"the lattice value {lattice_harmonic}."
            )
        elements = mapping.get("MachineElements", {})
        for role, key in (("normal_quadrupole", "normal_quadrupole_ords"),
                          ("skew_quadrupole", "skew_quadrupole_ords")):
            selected = list(elements.get(key, []))
            rows = grouped.get(role, [])
            valid = {
                int(row["lattice_ordinal"])
                for row in rows
                if row.get("lattice_ordinal") is not None
                and 0 <= int(row["lattice_ordinal"]) < len(ring)
                and str(getattr(ring[int(row["lattice_ordinal"])], "FamName", ""))
                == str(row.get("lattice_fam_name", ""))
            }
            invalid = [ordinal for ordinal in selected if ordinal not in valid]
            if invalid:
                errors.append(
                    f"Stage {index} {role} AT ordinal {invalid[0]} is not backed by "
                    "the saved Machine Mapping and active lattice."
                )

    return {
        "compatible": not errors,
        "errors": errors,
        "lattice": {
            "path": str(lattice_path),
            "sha256": file_sha256(lattice_path),
            "mat_variable": request.lattice_mat_variable,
            "element_count": len(ring),
            "rf_frequency_hz": lattice_frequency,
            "harmonic_number": lattice_harmonic,
        },
        "measurements": payload,
        "inventory": {"bpms": len(bpms), "horizontal_correctors": len(horizontal),
                      "vertical_correctors": len(vertical)},
        "stages": [stage.name for stage in recipe.stages],
    }


def smoke_recipe(recipe: FitRecipe) -> FitRecipe:
    """Shorten optimizer loops without changing inputs, ordering, or fit blocks."""
    result = copy.deepcopy(recipe)
    result.name += " — smoke test"
    for stage in result.stages:
        mapping = stage.configuration.get("backend_mapping", stage.configuration)
        options = mapping["LOCOOptions"]
        options["nIter"] = 1
        options["nLMIter"] = 1
        options["show_svd_plot"] = False
    return result


def validate_resume_session(
    session: FitRunSession, request: LocoRunRequest, recipe: FitRecipe
) -> list[str]:
    """Fail closed before reusing any completed workflow checkpoint."""
    errors: list[str] = []
    if session.recipe != recipe.to_dict():
        errors.append("Saved workflow session uses a different FIT recipe.")
    if session.lattice_checksum != file_sha256(request.lattice_path):
        errors.append("Saved workflow session belongs to a different reference lattice.")
    if session.lattice_mat_variable != request.lattice_mat_variable:
        errors.append("Saved workflow session uses a different MAT lattice variable.")
    expected_measurements = {
        role: str(Path(path).expanduser().resolve())
        for role, path in request.measurements.items()
    }
    saved_measurements = {
        role: str(Path(path).expanduser().resolve())
        for role, path in session.measurements.items()
    }
    if saved_measurements != expected_measurements:
        errors.append("Saved workflow session uses different measurement files.")
    for checkpoint in session.checkpoints:
        missing = checkpoint.validate_files()
        if missing:
            errors.append(
                f"Stage {checkpoint.stage_index + 1} checkpoint is incomplete: "
                + ", ".join(missing)
            )
    return errors


def _rms(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values * values))) if values.size else None


def stage_comparison(session: FitRunSession) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint in session.checkpoints:
        root = Path(checkpoint.results_dir)
        summary = json.loads((root / checkpoint.summary_file).read_text(encoding="utf-8"))
        metrics = (summary.get("iteration_metrics") or [{}])[-1]
        row: dict[str, Any] = {
            "stage": checkpoint.stage_index + 1,
            "name": checkpoint.stage_name,
            "results_dir": str(root),
            "chi2": (summary.get("chi2_history") or [None])[-1],
            "orm_residual_rms_m": (metrics.get("orm_residual") or {}).get("rms"),
            "svd_available": False,
        }
        optics = root / "optics_results.npz"
        if optics.is_file():
            with np.load(optics, allow_pickle=False) as archive:
                row["beta_x_beating_rms"] = _rms(archive["beta_beating_x"])
                row["beta_y_beating_rms"] = _rms(archive["beta_beating_y"])
        results = root / "loco_results.npz"
        if results.is_file():
            with np.load(results, allow_pickle=False) as archive:
                values = np.asarray(archive["fit_results"], dtype=float)
                delta = values[-1] - values[0] if values.ndim == 2 and len(values) > 1 else np.zeros(values.shape[-1])
                row["parameter_change_l2"] = float(np.linalg.norm(delta))
                row["parameter_change_max_abs"] = float(np.max(np.abs(delta))) if delta.size else 0.0
                row["svd_available"] = "singular_values" in archive.files
        rows.append(row)
    return rows


def _print_comparison(rows: list[dict[str, Any]]) -> None:
    print("\nFinal stage comparison")
    print("stage  name                          chi2          ORM RMS [m]   beta-x RMS   beta-y RMS   |Δp|₂")
    for row in rows:
        def number(key: str) -> str:
            value = row.get(key)
            return "n/a" if value is None else f"{float(value):.5g}"
        print(
            f"{row['stage']:>5}  {row['name'][:28]:<28}  {number('chi2'):>12}  "
            f"{number('orm_residual_rms_m'):>12}  {number('beta_x_beating_rms'):>10}  "
            f"{number('beta_y_beating_rms'):>10}  {number('parameter_change_l2'):>10}"
        )


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    project = ProjectMetadata.load(args.project.expanduser().resolve())
    recipe = FitRecipe.load(args.recipe.expanduser().resolve())
    if args.smoke_test:
        recipe = smoke_recipe(recipe)

    report = scientific_preflight(project, recipe)
    display_report = {key: value for key, value in report.items() if key != "measurements"}
    display_report["measurement_files"] = {
        role: {
            "path": entry.get("path"), "sha256": entry.get("sha256"),
            "shape": entry.get("shape"), "unit": entry.get("unit"),
            **({"rf_step_hz": entry.get("rf_step_hz"),
                "source_rf_sign": entry.get("source_rf_difference_sign_convention"),
                "canonical_rf_sign": entry.get("rf_difference_sign_convention")}
               if role == "dispersion" else {}),
        }
        for role, entry in (report.get("measurements", {}).get("files", {})).items()
    }
    print(json.dumps(display_report, indent=2))
    if not report["compatible"]:
        raise SystemExit("EBS MDT preflight failed; no fit was started.")
    if args.preflight_only:
        print("EBS MDT inputs are compatible. No fit was started.")
        return 0

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    project.loco_config.output_directory = str(output)
    request = LocoRunRequest.from_project(project)
    session_path = args.session.expanduser().resolve()
    resume_session = None
    if args.resume:
        if not session_path.is_file():
            raise SystemExit(f"Cannot resume; workflow session is absent: {session_path}")
        resume_session = FitRunSession.load(session_path)
        resume_errors = validate_resume_session(resume_session, request, recipe)
        if resume_errors:
            raise SystemExit("Cannot resume safely:\n- " + "\n- ".join(resume_errors))
    elif session_path.exists():
        raise SystemExit(
            f"Workflow session already exists: {session_path}\n"
            "Use --resume or select a different --session path."
        )

    def runner(stage_request, **callbacks):
        return run_loco_request(
            stage_request,
            log_callback=callbacks.get("log_callback"),
            progress_callback=callbacks.get("progress_callback"),
        )

    session = execute_workflow(
        request,
        recipe,
        session_path=session_path,
        runner=runner,
        log_callback=print,
        progress_callback=lambda event: print(json.dumps(event, sort_keys=True)),
        resume_session=resume_session,
    )
    comparison = stage_comparison(session)
    comparison_path = session_path.with_name(session_path.stem + "-comparison.json")
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    _print_comparison(comparison)
    print(f"\nWorkflow session: {session_path}")
    print(f"Comparison JSON: {comparison_path}")
    print(f"Completed {len(session.checkpoints)}/{len(recipe.stages)} stages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
