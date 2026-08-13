#!/usr/bin/env python3
"""Run pyLOCO for any configured measured accelerator machine."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pyLOCO.measured_machine.diagnostics import make_diagnostic_plots, save_run_results
from pyLOCO.measured_machine.workflow import (
    model_orm, output_directory, prepare_measurement, print_summary, run_fit,
)


def fit_modes(fit: dict) -> tuple[bool, bool]:
    coupling = any(name in fit["fit_list"] for name in (
        "skew_quads", "quads_tilt", "hbpm_coupling", "vbpm_coupling",
        "hcor_coupling", "vcor_coupling"))
    return coupling, fit["constraint_cfg"] is not None


def main(config_path: Path, *, prepare_only: bool = False) -> None:
    data = prepare_measurement(config_path)
    machine = data["cfg"].get("machine", {}).get("name", "measured machine")
    print(f"Prepared {machine}: {len(data['bpms'])} BPMs, "
          f"{len(data['correctors'][0])} H correctors, "
          f"{len(data['correctors'][1])} V correctors, ORM {data['orm'].shape}")
    if prepare_only:
        return
    started = time.perf_counter()
    initial_orm = model_orm(data)
    fit = run_fit(data)
    coupling, constrained = fit_modes(fit)
    output = output_directory(data, coupling=coupling, constrained=constrained)
    save_run_results(data, initial_orm, fit, output)
    make_diagnostic_plots(data, initial_orm, fit, output, coupling=coupling)
    fit["runtime_seconds"] = time.perf_counter() - started
    save_run_results(data, initial_orm, fit, output)
    print_summary(data, initial_orm, fit, coupling=coupling, constrained=constrained)
    print(f"Figures/results    : {output}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=here / "configs" / "petra_iii.yaml")
    parser.add_argument("--prepare-only", action="store_true",
                        help="validate and prepare inputs without calculating or fitting an ORM")
    arguments = parser.parse_args()
    main(arguments.config.resolve(), prepare_only=arguments.prepare_only)
