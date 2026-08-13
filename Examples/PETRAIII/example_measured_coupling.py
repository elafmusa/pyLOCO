#!/usr/bin/env python3
"""Fit the coupling blocks of a measured PETRA III ORM with pyLOCO."""
from pathlib import Path
import argparse
import time

from pyLOCO.measured_machine.diagnostics import make_diagnostic_plots, save_run_results
from pyLOCO.measured_machine.workflow import fit_modes, model_orm, output_directory, prepare_measurement, print_summary, run_fit


HERE = Path(__file__).resolve().parent


def main(config_path: Path) -> None:
    started = time.perf_counter()
    data = prepare_measurement(config_path)
    initial_orm = model_orm(data)
    fit = run_fit(data)
    coupling, constrained = fit_modes(fit)
    if not coupling:
        raise ValueError("The coupling example requires coupling parameter groups enabled in YAML")
    output = output_directory(data, coupling=coupling, constrained=constrained)
    save_run_results(data, initial_orm, fit, output)
    make_diagnostic_plots(data, initial_orm, fit, output, coupling=coupling)
    fit["runtime_seconds"] = time.perf_counter() - started
    save_run_results(data, initial_orm, fit, output)
    print_summary(data, initial_orm, fit, coupling=coupling, constrained=constrained)
    print(f"Figures            : {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "pyloco_config_coupling.yaml")
    main(parser.parse_args().config.resolve())
