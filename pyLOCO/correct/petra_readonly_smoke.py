"""Explicit hardware opt-in PETRA quadrupole diagnostics; zero writes."""
from __future__ import annotations

import argparse

from pyLOCO.control_system.petra import PETRAReadOnlyAdapter

from .model import load_review
from .petra_readonly import PETRACorrectReadOnlyService,apply_explicit_mapping,load_mapping,load_name_set


def main(argv=None):
    parser=argparse.ArgumentParser(description="PETRA read-only pyLOCO Correct smoke test (zero writes)")
    parser.add_argument("--results",required=True,help="pyLOCO Results directory or correction plan")
    parser.add_argument("--mapping",required=True,help="Explicit lattice-to-PETRA mapping JSON/YAML")
    parser.add_argument("--magnets",type=int,default=3,help="Maximum mapped normal quadrupoles to inspect")
    parser.add_argument("--sign-differences")
    parser.add_argument("--large-differences")
    args=parser.parse_args(argv)
    review=load_review(args.results); counts=apply_explicit_mapping(review,load_mapping(args.mapping)); mapped=[item for item in review.items if item.correction_type=="normal_quadrupole" and item.metadata.get("mapping_status")=="mapped"]
    keep={item.index for item in mapped[:max(0,args.magnets)]}
    for item in review.items:item.included=item.index in keep
    service=PETRACorrectReadOnlyService(PETRAReadOnlyAdapter(),sign_difference_names=load_name_set(args.sign_differences) if args.sign_differences else (),large_difference_names=load_name_set(args.large_differences) if args.large_differences else ())
    snapshot=service.read_snapshot(review,mapping_file=args.mapping)
    print(f"Mapping: {counts}")
    for row in snapshot.magnets:
        if row["index"] not in keep:continue
        print(f"{row['lattice_name']} -> {row['control_name']}: K={row['machine_k']}, I={row['current_ampere']}, target K={row['target_k']}, target I={row['target_current_ampere']}, limits=({row['min_current_ampere']}, {row['max_current_ampere']}), calibration={row['calibration_status']}, limit={row['current_limit_status']}")
    print("Completed PETRA read-only correction smoke test; zero writes were issued.")


if __name__=="__main__":main()
