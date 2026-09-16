#!/usr/bin/env python3
"""Run the historical ESRF-EBS Case C strategy through the current FIT backend.

The GUI project contains the lattice, device ordering, measurement conventions,
and four-stage recipe. Measurement paths may be replaced on the command line;
the scientific configuration and continuation semantics remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from pyLOCO.gui.backend import LocoRunRequest, run_loco_request
from pyLOCO.gui.fit_workflow import FitRecipe, FitRunSession, execute_workflow, preflight_recipe
from pyLOCO.gui.models.project import ProjectMetadata


HERE = Path(__file__).resolve().parent
DEFAULT_PROJECT = HERE / "EBS_case_C_multistage.pyloco.json"
DEFAULT_RECIPE = HERE / "recipe.json"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--orm", type=Path, help="replacement ORM HDF5 file")
    parser.add_argument("--dispersion", type=Path, help="replacement dispersion HDF5 file")
    parser.add_argument("--bpm-noise", type=Path, help="replacement BPM-noise HDF5 file")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--session", type=Path, default=HERE / "case_c_fit_session.json")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="continue an existing session")
    return parser.parse_args()


def _replace_measurements(project: ProjectMetadata, args: argparse.Namespace) -> None:
    replacements = {"orm": args.orm, "dispersion": args.dispersion, "bpm_noise": args.bpm_noise}
    for role, replacement in replacements.items():
        if replacement is None:
            continue
        source = replacement.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Replacement {role} file does not exist: {source}")
        project.measurements[role].path = str(source)
        project.measurements[role].size_bytes = source.stat().st_size


def main() -> int:
    args = _arguments()
    project = ProjectMetadata.load(args.project)
    recipe = FitRecipe.load(args.recipe)
    _replace_measurements(project, args)
    problems = project.validation_messages()
    if problems:
        raise SystemExit("Project validation failed:\n- " + "\n- ".join(problems))

    project.loco_config.output_directory = str(args.output.expanduser().resolve())
    request = LocoRunRequest.from_project(project)
    import at

    lattice = at.load_lattice(request.lattice_path, use="betamodel")
    names = [
        str(getattr(element, "CommonName", None) or getattr(element, "FamName", None)
            or getattr(element, "Name", ""))
        for element in lattice
    ]
    report = preflight_recipe(
        recipe,
        lattice_path=request.lattice_path,
        measurement_identity=request.measurement_session,
        lattice_element_names=names,
    )
    print(json.dumps(report, indent=2))
    if not report["compatible"]:
        raise SystemExit("Case C preflight failed.")
    if args.preflight_only:
        print("Case C project and four-stage recipe are compatible.")
        return 0

    session_path = args.session.expanduser().resolve()
    resume_session = None
    if args.resume:
        if not session_path.is_file():
            raise SystemExit(f"Cannot resume; session does not exist: {session_path}")
        resume_session = FitRunSession.load(session_path)
    elif session_path.exists():
        raise SystemExit(
            f"Session already exists: {session_path}\n"
            "Use --resume or choose a different --session path."
        )

    def runner(stage_request, **callbacks):
        return run_loco_request(
            stage_request,
            log_callback=callbacks.get("log_callback"),
            progress_callback=callbacks.get("progress_callback"),
        )

    session = execute_workflow(
        request,
        copy.deepcopy(recipe),
        session_path=session_path,
        runner=runner,
        log_callback=print,
        progress_callback=lambda event: print(json.dumps(event, sort_keys=True)),
        resume_session=resume_session,
    )
    print(f"Completed {len(session.checkpoints)}/{len(recipe.stages)} stages")
    print(f"Session: {session_path}")
    print(f"Final results: {session.checkpoints[-1].results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
