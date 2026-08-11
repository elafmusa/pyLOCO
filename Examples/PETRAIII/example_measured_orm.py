#!/usr/bin/env python3
"""Fit a measured PETRA III orbit-response matrix with pyLOCO."""
from pathlib import Path
import argparse

from petra_workflow import HERE, make_plots, model_orm, prepare_measurement, print_summary, run_fit


def main(config_path: Path) -> None:
    data = prepare_measurement(config_path)
    initial_orm = model_orm(data)
    fit = run_fit(data, coupling=False)
    output = make_plots(data, initial_orm, fit, coupling=False)
    print_summary(data, initial_orm, fit, coupling=False)
    print(f"Figures            : {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "pyloco_config.yaml")
    main(parser.parse_args().config.resolve())

