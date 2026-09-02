#!/usr/bin/env python3
"""Start the pyLOCO Monday-demo pySC control-system server.

The machine is the repository's existing ESRF-EBS SimulatedCommissioning
example.  The device catalog is generated from the exact SC object served, so
GUI selection order cannot drift from the orbit/control-system order.
"""
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
MACHINE = REPOSITORY / "Examples" / "EBS_pySC" / "com_simu_loco_example"
DEFAULT_CATALOG = HERE / "pysc_demo_catalog.json"
DEFAULT_BPM_NOISE_SIGMA_X_M = 1.5e-6
DEFAULT_BPM_NOISE_SIGMA_Y_M = 1.5e-6


def build_machine():
    from pySC import SimulatedCommissioning, disable_pySC_rich

    disable_pySC_rich()
    return SimulatedCommissioning.from_json(
        str(MACHINE / "ESRF_pySC_state.json"),
        lattice_file=str(MACHINE / "betamodel.mat"),
    )


def configure_demo_bpm_noise(
    sc,
    *,
    sigma_x_m: float = DEFAULT_BPM_NOISE_SIGMA_X_M,
    sigma_y_m: float = DEFAULT_BPM_NOISE_SIGMA_Y_M,
) -> None:
    """Override only the served demo object's BPM noise; never edit the EBS state."""
    if not np.isfinite(sigma_x_m) or sigma_x_m < 0:
        raise ValueError("Demo horizontal BPM noise must be finite and non-negative")
    if not np.isfinite(sigma_y_m) or sigma_y_m < 0:
        raise ValueError("Demo vertical BPM noise must be finite and non-negative")
    sc.bpm_system.noise_co_x = np.full(len(sc.bpm_system.names), sigma_x_m)
    sc.bpm_system.noise_co_y = np.full(len(sc.bpm_system.names), sigma_y_m)


def catalog_for(sc, *, host: str, port: int) -> dict:
    import at
    noise_x = np.asarray(sc.bpm_system.noise_co_x, dtype=float)
    noise_y = np.asarray(sc.bpm_system.noise_co_y, dtype=float)
    design_ring_4d = sc.lattice.ring.disable_6d(copy=True)
    momentum_compaction = float(at.get_mcf(design_ring_4d))
    inverse_gamma_squared = float(1.0 / design_ring_4d.gamma**2)
    # AT defines get_slip_factor() as 1/gamma^2 - alpha_c.  The RF
    # frequency relation used by Measure names eta with the accelerator-
    # physics convention alpha_c - 1/gamma^2, so retain both explicitly.
    at_slip_factor = float(design_ring_4d.get_slip_factor())
    eta = -at_slip_factor

    def common_noise(values: np.ndarray):
        return float(values[0]) if values.size and np.all(values == values[0]) else None

    return {
        "schema_version": "1.0",
        "backend": "pysc",
        "machine": "ESRF-EBS pySC demo",
        "host": host,
        "port": int(port),
        "rf_system": "main",
        "orbit_unit": "m",
        "corrector_unit": "rad",
        "rf_frequency_unit": "Hz",
        "metadata": {
            "configured_bpm_noise_sigma_x_m": common_noise(noise_x),
            "configured_bpm_noise_sigma_y_m": common_noise(noise_y),
            "corrector_control_unit": "rad",
            "momentum_compaction_factor": momentum_compaction,
            "relativistic_correction_inverse_gamma_squared": inverse_gamma_squared,
            "slip_factor": at_slip_factor,
            "slip_factor_convention": "Accelerator Toolbox get_slip_factor: 1/gamma^2 - alpha_c",
            "eta_alpha_minus_inverse_gamma_squared": eta,
            "eta_convention": "eta = alpha_c - 1/gamma^2 = -ring.get_slip_factor()",
            "momentum_relation": "delta_p_over_p = -(f - f0) / (eta * f0), eta = alpha_c - 1/gamma^2, first-order",
            "corrector_control_convention": "B1L/A1L are integrated normalized dipole strengths K0L, numerically interpreted as steering kick in radians; EBS B1L control sign is inverted by configuration",
        },
        "bpms": [str(name) for name in sc.bpm_system.names],
        "horizontal_correctors": [str(name) for name in sc.tuning.HCORR],
        "vertical_correctors": [str(name) for name in sc.tuning.VCORR],
    }


def install_server_compatibility_shim() -> None:
    """Fix the 1.5.2 ORBIT dispatch fall-through without editing site-packages.

    The released handler sends RAW/X or RAW/Y successfully and then falls into
    an unrelated ``else``.  The released server also caches one orbit for its
    refresh interval.  Here each new X/Y request pair gets one fresh capture,
    so BPM-noise samples are independent while both planes remain paired.
    Writes also need an immediate refresh so the next orbit is a deterministic
    readback.
    """
    import pySC.control_system.server as server
    from pySC.control_system.send_receive import pySCServerError, send_int, send_nparray

    original_magnet = server.magnet_server
    original_rf = server.rf_server
    orbit_pair = None
    planes_served = set()

    def orbit_handler(conn, signal, orbit_x, orbit_y, sc):
        nonlocal orbit_pair, planes_served
        variable = signal.split(" ")[1]
        command = signal[:3]
        if variable == "ORBIT/RAW/X" and command == "GET":
            if orbit_pair is None or "X" in planes_served:
                orbit_pair = sc.bpm_system.capture_orbit()
                planes_served = set()
            send_int(conn, 4); send_nparray(conn, np.asarray(orbit_pair[0]))
            planes_served.add("X")
            return
        if variable == "ORBIT/RAW/Y" and command == "GET":
            if orbit_pair is None or "Y" in planes_served:
                orbit_pair = sc.bpm_system.capture_orbit()
                planes_served = set()
            send_int(conn, 4); send_nparray(conn, np.asarray(orbit_pair[1]))
            planes_served.add("Y")
            if planes_served == {"X", "Y"}:
                orbit_pair = None
                planes_served = set()
            return
        if variable == "ORBIT/INJECTION/CORRECT" and command == "SET":
            sc.tuning.correct_injection(parameter=float(signal.split(" ")[2])); return
        raise pySCServerError

    def refresh_after_write(handler):
        def wrapped(conn, signal, sc):
            handler(conn, signal, sc)
            if signal.startswith("SET "):
                # The reply has already been sent.  Re-entering the server's
                # outer loop refreshes orbit before it accepts the next client.
                raise socket.timeout
        return wrapped

    server.orbit_server = orbit_handler
    server.magnet_server = refresh_after_write(original_magnet)
    server.rf_server = refresh_after_write(original_rf)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=13131)
    parser.add_argument("--refresh-rate", type=float, default=1.0)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--bpm-noise-x-um", type=float, default=DEFAULT_BPM_NOISE_SIGMA_X_M*1e6,
                        help="Demo horizontal BPM-noise sigma in micrometres (default: 1.5)")
    parser.add_argument("--bpm-noise-y-um", type=float, default=DEFAULT_BPM_NOISE_SIGMA_Y_M*1e6,
                        help="Demo vertical BPM-noise sigma in micrometres (default: 1.5)")
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1":
        parser.error("The installed pySC server binds to 127.0.0.1 only")

    sc = build_machine()
    configure_demo_bpm_noise(
        sc,
        sigma_x_m=args.bpm_noise_x_um*1e-6,
        sigma_y_m=args.bpm_noise_y_um*1e-6,
    )
    catalog = catalog_for(sc, host=args.host, port=args.port)
    args.catalog.parent.mkdir(parents=True, exist_ok=True)
    args.catalog.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(f"pySC demo machine: {catalog['machine']}")
    print(f"BPMs: {len(catalog['bpms'])}; H correctors: {len(catalog['horizontal_correctors'])}; V correctors: {len(catalog['vertical_correctors'])}")
    print(f"Demo BPM noise: σx={args.bpm_noise_x_um:g} µm; σy={args.bpm_noise_y_um:g} µm")
    print(f"Catalog: {args.catalog.resolve()}")
    print(f"Listening on {args.host}:{args.port} (Ctrl-C to stop)")

    install_server_compatibility_shim()
    from pySC.control_system.server import start_server
    start_server(sc, port=args.port, refresh_rate=args.refresh_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
