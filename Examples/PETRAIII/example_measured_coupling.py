#!/usr/bin/env python3
"""Fit the coupling blocks of a measured PETRA III ORM with pyLOCO."""
from pathlib import Path
import argparse

from petra_workflow import HERE, make_plots, model_orm, prepare_measurement, print_summary, run_fit


def main(config_path: Path) -> None:
    data = prepare_measurement(config_path)
    initial_orm = model_orm(data)
    fit = run_fit(data, coupling=True)
    output = make_plots(data, initial_orm, fit, coupling=True)
    print_summary(data, initial_orm, fit, coupling=True)
    print(f"Figures            : {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "pyloco_config.yaml")
    main(parser.parse_args().config.resolve())

