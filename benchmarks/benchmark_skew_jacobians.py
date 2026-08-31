#!/usr/bin/env python3
"""Compare legacy, vectorized, and numerical skew-quadrupole Jacobians.

All element selections are comma-separated AT ordinals, making this utility
independent of PETRA-IV-specific setup and suitable for smaller lattices.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import at
import numpy as np

from pyLOCO.analytic_orm_with_skew_quad_errors import (
    analytic_orm_variation_with_skew_quadrupole,
)
from pyLOCO.config import RMConfig
from pyLOCO.response_matrix import response_matrix


def _indices(value):
    return np.asarray([int(item) for item in value.split(",") if item.strip()], dtype=int)


def _analytical(ring, bpms, hcors, vcors, skews, implementation, *,
                thick_skews=True, thick_steerers=False, kick=1e-5):
    started = time.perf_counter()
    yx, _ = analytic_orm_variation_with_skew_quadrupole(
        ring, bpms, hcors, skews, verbose=False, thick_skew=thick_skews,
        thick_steerer=thick_steerers, implementation=implementation,
    )
    _, xy = analytic_orm_variation_with_skew_quadrupole(
        ring, bpms, vcors, skews, verbose=False, thick_skew=thick_skews,
        thick_steerer=thick_steerers, implementation=implementation,
    )
    result = np.zeros((len(skews), 2 * len(bpms), len(hcors) + len(vcors)))
    lengths = np.asarray([ring[int(index)].Length for index in skews], dtype=float)
    result[:, len(bpms):, :len(hcors)] = (
        -np.moveaxis(yx, 2, 0) * lengths[:, None, None] * kick
    )
    result[:, :len(bpms), len(hcors):] = (
        np.moveaxis(xy, 2, 0) * lengths[:, None, None] * kick
    )
    return result, time.perf_counter() - started


def _numerical(ring, bpms, hcors, vcors, skews, step, *, kick=1e-5,
               dispersion=False, cavities=None, rf_step=-3000.0):
    started = time.perf_counter()
    config = RMConfig(
        bpm_ords=bpms, cm_ords=(hcors, vcors), cav_ords=cavities,
        dkick=(np.full(len(hcors), kick), np.full(len(vcors), kick)),
        includeDispersion=dispersion, rfStep=rf_step, calculator="Linear",
    )
    values = []
    for ordinal in skews:
        plus, minus = ring.deepcopy(), ring.deepcopy()
        plus[int(ordinal)].PolynomA[1] += step
        minus[int(ordinal)].PolynomA[1] -= step
        values.append(
            (response_matrix(plus, config=config) - response_matrix(minus, config=config))
            / (2.0 * step)
        )
    return np.asarray(values), time.perf_counter() - started


def _metrics(name, value, seconds, reference, reference_seconds):
    difference = value - reference
    reference_norm = np.linalg.norm(reference)
    print(
        f"{name:10s} runtime={seconds:.6g}s "
        f"speedup={reference_seconds / seconds:.6g} shape={value.shape} "
        f"RMS={np.sqrt(np.mean(value**2)):.6g} "
        f"max_abs={np.max(np.abs(value)):.6g} "
        f"relative_norm_difference={np.linalg.norm(difference) / reference_norm:.6g} "
        f"RMS_difference={np.sqrt(np.mean(difference**2)):.6g}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lattice")
    parser.add_argument("--bpms", required=True)
    parser.add_argument("--hcors", required=True)
    parser.add_argument("--vcors", required=True)
    parser.add_argument("--skews", required=True)
    parser.add_argument("--step", type=float, default=1e-6)
    parser.add_argument("--kick", type=float, default=1e-5)
    parser.add_argument("--thin-skews", action="store_true")
    parser.add_argument("--thick-steerers", action="store_true")
    parser.add_argument("--dispersion", action="store_true")
    parser.add_argument("--cavities")
    parser.add_argument("--rf-step", type=float, default=-3000.0)
    args = parser.parse_args(argv)

    ring = at.load_lattice(args.lattice)
    bpms, hcors, vcors, skews = (
        _indices(getattr(args, name)) for name in ("bpms", "hcors", "vcors", "skews")
    )
    cavities = _indices(args.cavities) if args.cavities else None
    if args.dispersion and cavities is None:
        parser.error("--dispersion requires --cavities")

    common = dict(
        thick_skews=not args.thin_skews,
        thick_steerers=args.thick_steerers,
        kick=args.kick,
    )
    legacy, legacy_seconds = _analytical(
        ring, bpms, hcors, vcors, skews, "legacy", **common
    )
    vectorized, vectorized_seconds = _analytical(
        ring, bpms, hcors, vcors, skews, "vectorized", **common
    )
    numerical, numerical_seconds = _numerical(
        ring, bpms, hcors, vcors, skews, args.step, kick=args.kick,
        dispersion=args.dispersion, cavities=cavities, rf_step=args.rf_step,
    )
    if args.dispersion:
        # Analytical skew formulas model the ORM cross-plane blocks. Append
        # the shared numerical dispersion derivative for like-for-like shape.
        dispersion_column = numerical[:, :, -1:]
        legacy = np.concatenate((legacy, dispersion_column), axis=2)
        vectorized = np.concatenate((vectorized, dispersion_column), axis=2)
    _metrics("legacy", legacy, legacy_seconds, numerical, numerical_seconds)
    _metrics("vectorized", vectorized, vectorized_seconds, numerical, numerical_seconds)
    _metrics("numerical", numerical, numerical_seconds, numerical, numerical_seconds)


if __name__ == "__main__":
    main()
