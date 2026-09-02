#!/usr/bin/env python3
"""Destructive-but-restoring protocol smoke test for the local pySC demo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=HERE / "pysc_demo_catalog.json")
    parser.add_argument("--kick", type=float, default=1e-6)
    parser.add_argument("--rf-step", type=float, default=20.0)
    args = parser.parse_args(argv)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))

    from pySC.control_system.client import read, write
    base = f"{catalog['host']}:{catalog['port']}"
    address = lambda variable: f"{base}/{variable}"
    orbit = lambda: (
        np.asarray(read(address("ORBIT/RAW/X")), dtype=float),
        np.asarray(read(address("ORBIT/RAW/Y")), dtype=float),
    )

    x0, y0 = orbit()
    if x0.shape != y0.shape or x0.size != len(catalog["bpms"]):
        raise RuntimeError("Orbit arrays do not match the generated BPM catalog")

    hcor = catalog["horizontal_correctors"][0]
    h_address = address(f"MAGNET/{hcor}")
    h0 = float(read(h_address))
    try:
        write(h_address, h0 + args.kick)
        h1 = float(read(h_address)); xh, yh = orbit()
        if h1 == h0 or np.array_equal(xh, x0) and np.array_equal(yh, y0):
            raise RuntimeError("Corrector change produced no setpoint/orbit change")
    finally:
        write(h_address, h0)
    if float(read(h_address)) != h0:
        raise RuntimeError("Exact corrector restoration failed")

    rf_address = address(f"RF/{catalog['rf_system']}/FREQUENCY")
    rf0 = float(read(rf_address))
    try:
        write(rf_address, rf0 + args.rf_step); xp, yp = orbit()
        write(rf_address, rf0 - args.rf_step); xm, ym = orbit()
        if np.array_equal(xp, xm) and np.array_equal(yp, ym):
            raise RuntimeError("RF change produced no orbit change")
    finally:
        write(rf_address, rf0)
    if float(read(rf_address)) != rf0:
        raise RuntimeError("Exact RF restoration failed")

    print(f"PASS orbit: {x0.size} X + {y0.size} Y BPM values")
    print(f"PASS H corrector: {hcor}; restored exactly to {h0:.16g}")
    print(f"PASS RF/main: bipolar response observed; restored exactly to {rf0:.16g} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
