#!/usr/bin/env python3
"""Compare legacy, vectorized, and numerical normal-quadrupole Jacobians.

Indices are supplied as comma-separated AT ordinals, so the benchmark is
lattice-agnostic. The analytical implementations use identical thick/thin
options. Numerical results use a central difference of the Linear ORM.
"""

from __future__ import annotations

import argparse
import time

import at
import numpy as np

from pyLOCO.analytic_orm_with_normal_quad_errors import (
    analytic_orm_variation_with_normal_quadrupole,
)
from pyLOCO.config import RMConfig
from pyLOCO.response_matrix import calculate_rf_response, response_matrix


def _indices(value: str) -> np.ndarray:
    return np.asarray([int(item) for item in value.split(",") if item.strip()], dtype=int)


def _analytical(ring, bpms, hcors, vcors, quads, implementation, thick_quads, thick_steerers,
                *, dispersion=False, cavities=None, rf_step=-3000.0, step=1e-6):
    started = time.perf_counter()
    dh, _ = analytic_orm_variation_with_normal_quadrupole(
        ring, bpms, hcors, quads, verbose=False,
        thick_quadrupole=thick_quads, thick_steerers=thick_steerers,
        implementation=implementation,
    )
    _, dv = analytic_orm_variation_with_normal_quadrupole(
        ring, bpms, vcors, quads, verbose=False,
        thick_quadrupole=thick_quads, thick_steerers=thick_steerers,
        implementation=implementation,
    )
    result = np.zeros((len(quads), 2 * len(bpms), len(hcors) + len(vcors)))
    result[:, :len(bpms), :len(hcors)] = np.moveaxis(dh, 2, 0)
    result[:, len(bpms):, len(hcors):] = np.moveaxis(dv, 2, 0)
    if dispersion:
        eta = []
        config = RMConfig(bpm_ords=bpms, cm_ords=(hcors, vcors), cav_ords=cavities,
                          dkick=(1e-5, 1e-5), includeDispersion=True,
                          rfStep=rf_step, calculator="Linear")
        for ordinal in quads:
            plus, minus = ring.deepcopy(), ring.deepcopy()
            plus[int(ordinal)].PolynomB[1] += step
            minus[int(ordinal)].PolynomB[1] -= step
            eta.append((calculate_rf_response(plus, config=config) -
                        calculate_rf_response(minus, config=config)) / (2 * step))
        result = np.concatenate((result, np.asarray(eta)[:, :, None]), axis=2)
    return result, time.perf_counter() - started


def _numerical(ring, bpms, hcors, vcors, quads, step, *, dispersion=False,
               cavities=None, rf_step=-3000.0):
    started = time.perf_counter()
    values = []
    config = RMConfig(
        bpm_ords=bpms, cm_ords=(hcors, vcors), dkick=(1e-5, 1e-5),
        cav_ords=cavities, includeDispersion=dispersion, rfStep=rf_step,
        calculator="Linear",
    )
    for ordinal in quads:
        plus, minus = ring.deepcopy(), ring.deepcopy()
        plus[int(ordinal)].PolynomB[1] += step
        minus[int(ordinal)].PolynomB[1] -= step
        values.append((response_matrix(plus, config=config) - response_matrix(minus, config=config)) / (2 * step))
    return np.asarray(values), time.perf_counter() - started


def _metrics(name, value, seconds, reference, reference_seconds):
    difference = value - reference
    norm = np.linalg.norm(reference)
    print(f"{name:10s} runtime={seconds:.6g}s speedup={reference_seconds / seconds:.6g} "
          f"shape={value.shape} RMS={np.sqrt(np.mean(value**2)):.6g} "
          f"max_abs={np.max(np.abs(value)):.6g} "
          f"relative_norm_difference={np.linalg.norm(difference) / norm:.6g} "
          f"RMS_difference={np.sqrt(np.mean(difference**2)):.6g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lattice")
    parser.add_argument("--bpms", required=True)
    parser.add_argument("--hcors", required=True)
    parser.add_argument("--vcors", required=True)
    parser.add_argument("--quads", required=True)
    parser.add_argument("--step", type=float, default=1e-6)
    parser.add_argument("--dispersion", action="store_true")
    parser.add_argument("--cavities", help="Comma-separated RF cavity ordinals")
    parser.add_argument("--rf-step", type=float, default=-3000.0)
    parser.add_argument("--thin-quads", action="store_true")
    parser.add_argument("--thick-steerers", action="store_true")
    args = parser.parse_args()
    ring = at.load_lattice(args.lattice)
    selections = tuple(_indices(getattr(args, key)) for key in ("bpms", "hcors", "vcors", "quads"))
    cavities = _indices(args.cavities) if args.cavities else None
    common = dict(dispersion=args.dispersion, cavities=cavities,
                  rf_step=args.rf_step, step=args.step)
    if args.dispersion and cavities is None:
        parser.error("--dispersion requires --cavities")
    legacy, legacy_s = _analytical(ring, *selections, "legacy", not args.thin_quads,
                                   args.thick_steerers, **common)
    vectorized, vectorized_s = _analytical(ring, *selections, "vectorized", not args.thin_quads,
                                           args.thick_steerers, **common)
    numerical, numerical_s = _numerical(
        ring, *selections, args.step, dispersion=args.dispersion,
        cavities=cavities, rf_step=args.rf_step,
    )
    _metrics("legacy", legacy, legacy_s, numerical, numerical_s)
    _metrics("vectorized", vectorized, vectorized_s, numerical, numerical_s)
    _metrics("numerical", numerical, numerical_s, numerical, numerical_s)


if __name__ == "__main__":
    main()
