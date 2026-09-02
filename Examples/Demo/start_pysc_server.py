#!/usr/bin/env python3
"""Start a compatibility-protected pySC server for a named machine profile."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from pyLOCO.control_system.pysc_profiles import load_pysc_profile

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 13131


def _array_statistics(values) -> dict:
    values = np.asarray(values, dtype=float).ravel()
    return {
        "count": int(values.size), "mean": float(np.mean(values)),
        "rms": float(np.sqrt(np.mean(values**2))),
        "minimum": float(np.min(values)), "maximum": float(np.max(values)),
    }


def realized_error_statistics(sc) -> dict:
    """Return the actual fixed-seed draws applied to the served SC object."""
    support = sc.support_system.data["L0"]

    def element_values(category, field):
        return [getattr(support[sc.magnet_settings.magnets[name].sim_index], field)
                for name in sc.magnet_arrays[category]]

    def link_errors(category, component, *, inverted=False):
        values = []
        for name in sc.magnet_arrays[category]:
            link = sc.magnet_settings.links[f"{name}/{component}->{name}/{component}"]
            factor = abs(link.error.factor) if inverted else link.error.factor
            values.append(factor - 1.0)
        return values

    if not {"quadrupoles", "HCM", "VCM"}.issubset(sc.magnet_arrays):
        return {}
    return {
        "quadrupole_alignment_x_m": _array_statistics(element_values("quadrupoles", "dx")),
        "quadrupole_alignment_y_m": _array_statistics(element_values("quadrupoles", "dy")),
        "quadrupole_roll_rad": _array_statistics(element_values("quadrupoles", "roll")),
        "quadrupole_relative_calibration": _array_statistics(link_errors("quadrupoles", "B2")),
        "hcor_roll_rad": _array_statistics(element_values("HCM", "roll")),
        "vcor_roll_rad": _array_statistics(element_values("VCM", "roll")),
        "hcor_relative_calibration": _array_statistics(link_errors("HCM", "B1L", inverted=True)),
        "vcor_relative_calibration": _array_statistics(link_errors("VCM", "A1L")),
        "bpm_offset_x_m": _array_statistics(sc.bpm_system.offsets_x),
        "bpm_offset_y_m": _array_statistics(sc.bpm_system.offsets_y),
        "bpm_roll_rad": _array_statistics(sc.bpm_system.rolls),
        "bpm_relative_gain_x": _array_statistics(sc.bpm_system.calibration_errors_x),
        "bpm_relative_gain_y": _array_statistics(sc.bpm_system.calibration_errors_y),
        "bpm_noise_sigma_x_m": _array_statistics(sc.bpm_system.noise_co_x),
        "bpm_noise_sigma_y_m": _array_statistics(sc.bpm_system.noise_co_y),
    }


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def configure_demo_bpm_noise(sc, *, sigma_x_m: float, sigma_y_m: float) -> None:
    """Override only the served object's noise arrays, never its source state."""
    for plane, value in (("horizontal", sigma_x_m), ("vertical", sigma_y_m)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Demo {plane} BPM noise must be finite and non-negative")
    sc.bpm_system.noise_co_x = np.full(len(sc.bpm_system.names), sigma_x_m)
    sc.bpm_system.noise_co_y = np.full(len(sc.bpm_system.names), sigma_y_m)


def build_profile_machine(profile_key: str, *, repository_root: Path | None = None):
    """Build the SC exactly as declared by the selected profile manifest."""
    profile = load_pysc_profile(profile_key, repository_root=repository_root)
    builder = profile.configuration["builder"]
    if builder == "saved_state":
        from pySC import SimulatedCommissioning, disable_pySC_rich
        disable_pySC_rich()
        sc = SimulatedCommissioning.from_json(
            str(profile.resolve("state_file")), lattice_file=str(profile.resolve("lattice_file")),
        )
        configure_demo_bpm_noise(
            sc,
            sigma_x_m=float(profile.configuration["demo_bpm_noise_sigma_x_m"]),
            sigma_y_m=float(profile.configuration["demo_bpm_noise_sigma_y_m"]),
        )
    elif builder == "generated_configuration":
        from pySC import generate_SC
        with _working_directory(profile.directory):
            sc = generate_SC(
                profile.configuration["configuration_file"],
                seed=int(profile.configuration.get("random_seed", 1)),
                sigma_truncate=profile.configuration.get("sigma_truncate"),
            )
    else:
        raise RuntimeError(f"Unsupported pySC profile builder: {builder}")
    return profile, sc


def catalog_for(sc, profile, *, host: str, port: int) -> dict:
    """Generate inventory and lattice metadata from the exact served SC object."""
    import at
    noise_x = np.asarray(sc.bpm_system.noise_co_x, dtype=float)
    noise_y = np.asarray(sc.bpm_system.noise_co_y, dtype=float)
    ring = sc.lattice.ring.disable_6d(copy=True)
    alpha = float(at.get_mcf(ring))
    inverse_gamma_squared = float(1.0 / ring.gamma**2)
    at_slip = float(ring.get_slip_factor())

    def common_noise(values):
        return float(values[0]) if values.size and np.all(values == values[0]) else None

    catalog = {
        "schema_version": "1.1", "backend": "pysc", "profile": profile.key,
        "profile_label": profile.label, "scenario": profile.scenario, "machine": profile.machine,
        "host": host, "port": int(port), "rf_system": profile.configuration.get("rf_system", "main"),
        "orbit_unit": "m", "corrector_unit": "rad", "rf_frequency_unit": "Hz",
        "provenance": profile.configuration.get("provenance", {}),
        "metadata": {
            "machine_profile": profile.key, "machine_profile_label": profile.label,
            "machine_scenario": profile.scenario,
            "configured_bpm_noise_sigma_x_m": common_noise(noise_x),
            "configured_bpm_noise_sigma_y_m": common_noise(noise_y),
            "corrector_control_unit": "rad", "momentum_compaction_factor": alpha,
            "relativistic_correction_inverse_gamma_squared": inverse_gamma_squared,
            "slip_factor": at_slip,
            "slip_factor_convention": "Accelerator Toolbox get_slip_factor: 1/gamma^2 - alpha_c",
            "eta_alpha_minus_inverse_gamma_squared": -at_slip,
            "eta_convention": "eta = alpha_c - 1/gamma^2 = -ring.get_slip_factor()",
            "momentum_relation": "delta_p_over_p = -(f - f0) / (eta * f0), eta = alpha_c - 1/gamma^2, first-order",
            "corrector_control_convention": "B1L/A1L are integrated normalized dipole strengths; profile configuration owns sign conventions",
            "random_seed": profile.configuration.get("random_seed"),
            "sigma_truncate": profile.configuration.get("sigma_truncate"),
            "requested_error_distribution": profile.configuration.get("error_distribution"),
            "realized_error_statistics": realized_error_statistics(sc),
        },
        "bpms": [str(name) for name in sc.bpm_system.names],
        "horizontal_correctors": [str(name) for name in sc.tuning.HCORR],
        "vertical_correctors": [str(name) for name in sc.tuning.VCORR],
    }
    return catalog


def install_server_compatibility_shim() -> None:
    """Provide fresh paired orbits and immediate post-write refresh locally."""
    import pySC.control_system.server as server
    from pySC.control_system.send_receive import pySCServerError, send_int, send_nparray
    original_magnet, original_rf = server.magnet_server, server.rf_server
    orbit_pair = None
    planes_served: set[str] = set()

    def orbit_handler(conn, signal, orbit_x, orbit_y, sc):
        nonlocal orbit_pair, planes_served
        variable, command = signal.split(" ")[1], signal[:3]
        if variable == "ORBIT/RAW/X" and command == "GET":
            if orbit_pair is None or "X" in planes_served:
                orbit_pair, planes_served = sc.bpm_system.capture_orbit(), set()
            send_int(conn, 4); send_nparray(conn, np.asarray(orbit_pair[0])); planes_served.add("X"); return
        if variable == "ORBIT/RAW/Y" and command == "GET":
            if orbit_pair is None or "Y" in planes_served:
                orbit_pair, planes_served = sc.bpm_system.capture_orbit(), set()
            send_int(conn, 4); send_nparray(conn, np.asarray(orbit_pair[1])); planes_served.add("Y")
            if planes_served == {"X", "Y"}: orbit_pair, planes_served = None, set()
            return
        if variable == "ORBIT/INJECTION/CORRECT" and command == "SET":
            sc.tuning.correct_injection(parameter=float(signal.split(" ")[2])); return
        raise pySCServerError

    def refresh_after_write(handler):
        def wrapped(conn, signal, sc):
            handler(conn, signal, sc)
            if signal.startswith("SET "): raise socket.timeout
        return wrapped

    server.orbit_server = orbit_handler
    server.magnet_server = refresh_after_write(original_magnet)
    server.rf_server = refresh_after_write(original_rf)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("ebs", "petra3", "petra3_realistic"), required=True)
    parser.add_argument("--host", default=DEFAULT_HOST); parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--refresh-rate", type=float, default=1.0); parser.add_argument("--catalog", type=Path)
    parser.add_argument("--bpm-noise-x-um", type=float); parser.add_argument("--bpm-noise-y-um", type=float)
    args = parser.parse_args(argv)
    if args.host != DEFAULT_HOST: parser.error("The installed pySC server binds to 127.0.0.1 only")
    profile, sc = build_profile_machine(args.profile)
    if args.profile == "ebs" and (args.bpm_noise_x_um is not None or args.bpm_noise_y_um is not None):
        x = args.bpm_noise_x_um if args.bpm_noise_x_um is not None else float(sc.bpm_system.noise_co_x[0]) * 1e6
        y = args.bpm_noise_y_um if args.bpm_noise_y_um is not None else float(sc.bpm_system.noise_co_y[0]) * 1e6
        configure_demo_bpm_noise(sc, sigma_x_m=x * 1e-6, sigma_y_m=y * 1e-6)
    elif args.profile != "ebs" and (args.bpm_noise_x_um is not None or args.bpm_noise_y_um is not None):
        parser.error("BPM-noise overrides are available only for the configurable EBS demo profile")
    catalog = catalog_for(sc, profile, host=args.host, port=args.port)
    path = args.catalog.expanduser().resolve() if args.catalog else profile.catalog_path
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    legacy = profile.configuration.get("legacy_catalog_file")
    if legacy and args.catalog is None:
        legacy_path = (profile.directory / legacy).resolve()
        legacy_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    rf = sc.rf_settings.systems[catalog["rf_system"]].frequency
    print(f"pySC profile: {profile.label} / {profile.scenario}")
    print(f"BPMs: {len(catalog['bpms'])}; H correctors: {len(catalog['horizontal_correctors'])}; V correctors: {len(catalog['vertical_correctors'])}")
    print(f"RF system: {catalog['rf_system']}; frequency: {rf:.6f} Hz")
    print(f"Catalog: {path}"); print(f"Listening on {args.host}:{args.port} (Ctrl-C to stop)")
    install_server_compatibility_shim()
    from pySC.control_system.server import start_server
    start_server(sc, port=args.port, refresh_rate=args.refresh_rate)
    return 0


if __name__ == "__main__": raise SystemExit(main())
