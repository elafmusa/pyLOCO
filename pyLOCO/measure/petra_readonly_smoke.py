"""Explicit PETRA read-only smoke test. This module contains no write call."""
from __future__ import annotations

import argparse
from pathlib import Path

from pyLOCO.control_system import PETRAReadOnlyAdapter


def _names(path):
    return tuple(line.strip() for line in Path(path).read_text().splitlines() if line.strip())


def main(argv=None):
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description="Read-only PETRA/DOOCS connectivity smoke test (zero writes)")
    parser.add_argument("--bpms",default=str(root/"Examples/PETRAIII/data/BPM_names.txt"))
    parser.add_argument("--hcors",default=str(root/"Examples/PETRAIII/data/HCM_names_control.txt"))
    parser.add_argument("--vcors",default=str(root/"Examples/PETRAIII/data/VCM_names_control.txt"))
    parser.add_argument("--correctors",type=int,default=3,help="Number of correctors to read safely")
    parser.add_argument("--calibration",action="store_true",help="Also read current limits and test strength-to-current at the present KICK.SP")
    args=parser.parse_args(argv)
    adapter=PETRAReadOnlyAdapter(_names(args.bpms),_names(args.hcors),_names(args.vcors))
    x,y=adapter.read_orbit(); print(f"BPM orbit: X={x.size}, Y={y.size}; units converted nm -> m")
    names=(adapter.horizontal_corrector_names+adapter.vertical_corrector_names)[:max(0,args.correctors)]
    for name in names:
        values=adapter.read_corrector_diagnostics(name); print(name,", ".join(f"{key}={value:g}" for key,value in values.items()))
        if args.calibration:
            limits=adapter.current_limits(name); current=adapter.strength_to_current(name,values["KICK.SP"]); print(f"  calibration current={current:g}, limits={limits}")
    print("RF readback: unavailable (no verified channel mapping in repository)")
    print("Completed read-only smoke test; zero writes were issued.")


if __name__=="__main__":main()
