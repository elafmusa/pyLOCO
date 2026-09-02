#!/usr/bin/env python3
"""Backward-compatible launcher for the validated EBS pySC profile."""
from __future__ import annotations

from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from Examples.Demo.start_pysc_server import (
    build_profile_machine,
    catalog_for as _catalog_for,
    configure_demo_bpm_noise,
    install_server_compatibility_shim,
    main as _profile_main,
)
from pyLOCO.control_system.pysc_profiles import load_pysc_profile

HERE = Path(__file__).resolve().parent
DEFAULT_CATALOG = HERE / "pysc_demo_catalog.json"
DEFAULT_BPM_NOISE_SIGMA_X_M = 1.5e-6
DEFAULT_BPM_NOISE_SIGMA_Y_M = 1.5e-6


def build_machine():
    return build_profile_machine("ebs")[1]


def catalog_for(sc, *, host: str, port: int) -> dict:
    profile = load_pysc_profile("ebs")
    return _catalog_for(sc, profile, host=host, port=port)


def main(argv=None) -> int:
    values = sys.argv[1:] if argv is None else list(argv)
    return _profile_main(["--profile", "ebs", *values])


if __name__ == "__main__":
    raise SystemExit(main())
