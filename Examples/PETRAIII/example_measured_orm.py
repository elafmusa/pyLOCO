#!/usr/bin/env python3
"""Fit a measured PETRA III orbit-response matrix with pyLOCO."""
from pathlib import Path
import argparse
import time

from petra_workflow import HERE, make_diagnostic_plots, model_orm, output_directory, prepare_measurement, print_summary, run_fit, save_run_results


def main(config_path: Path) -> None:
    started = time.perf_counter()
    data = prepare_measurement(config_path)
    initial_orm = model_orm(data)
    fit = run_fit(data, coupling=False)
    output = output_directory(data, coupling=False, constrained=False)
    save_run_results(data, initial_orm, fit, output)
    make_diagnostic_plots(data, initial_orm, fit, output, coupling=False)
    fit["runtime_seconds"] = time.perf_counter() - started
    save_run_results(data, initial_orm, fit, output)
    print_summary(data, initial_orm, fit, coupling=False)
    print(f"Figures            : {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "pyloco_config.yaml")
    main(parser.parse_args().config.resolve())
