"""Read-only pre-FIT comparison of measured data with the reference lattice."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .backend import (
    LocoRunRequest, _apply_bad_bpm_positions, _apply_corrector_exclusions,
    _apply_machine_element_selections, _corrector_steps, _derive_indices,
    _load_bad_bpm_positions, _load_measurements, _make_gui_config,
    _prepare_loco_lattice, _validate_indices,
)
from .lattice_loader import load_project_lattice


@dataclass(frozen=True)
class ReferenceComparisonResult:
    measured_orm: np.ndarray
    model_orm: np.ndarray
    measured_rf: np.ndarray | None
    model_rf: np.ndarray | None
    physical_measured: np.ndarray | None
    physical_model: np.ndarray | None
    n_bpms: int
    n_hcors: int
    n_vcors: int
    cross_plane_excluded: bool
    statistics: dict[str, dict[str, float]]
    rf_statistics: dict[str, dict[str, float]]
    provenance: dict[str, Any]


def _rms(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite ** 2))) if finite.size else float("nan")


def _statistics(measured: np.ndarray, model: np.ndarray, nb: int, nh: int) -> dict[str, dict[str, float]]:
    residual = measured - model
    blocks = {
        "Global": residual,
        "H→X": residual[:nb, :nh],
        "V→X": residual[:nb, nh:],
        "H→Y": residual[nb:, :nh],
        "V→Y": residual[nb:, nh:],
    }
    return {
        name: {
            "measured_rms": _rms(measured if name == "Global" else {
                "H→X": measured[:nb, :nh], "V→X": measured[:nb, nh:],
                "H→Y": measured[nb:, :nh], "V→Y": measured[nb:, nh:],
            }[name]),
            "model_rms": _rms(model if name == "Global" else {
                "H→X": model[:nb, :nh], "V→X": model[:nb, nh:],
                "H→Y": model[nb:, :nh], "V→Y": model[nb:, nh:],
            }[name]),
            "residual_rms": _rms(values),
        }
        for name, values in blocks.items()
    }


def _rf_statistics(measured: np.ndarray | None, model: np.ndarray | None,
                   nb: int) -> dict[str, dict[str, float]]:
    if measured is None or model is None:
        return {}
    return {
        plane: {
            "measured_rms": _rms(measured[section]),
            "model_rms": _rms(model[section]),
            "residual_rms": _rms(measured[section] - model[section]),
        }
        for plane, section in (("Horizontal", slice(0, nb)),
                               ("Vertical", slice(nb, None)))
    }
def calculate_reference_comparison(
    request: LocoRunRequest,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> ReferenceComparisonResult:
    """Calculate only the reference ORM/RF response used by FIT.

    This deliberately stops before FIT initialization, Jacobian construction,
    constraints, SVD and optimization.  It performs no control-system access.
    """
    cancelled = cancelled or (lambda: False)
    if cancelled():
        raise RuntimeError("Reference-model comparison cancelled.")

    from pyLOCO.pyloco import remove_bad_bpms
    from pyLOCO.response_matrix import response_matrix

    ring = _prepare_loco_lattice(
        load_project_lattice(request.lattice_path, request.lattice_mat_variable or None)
    )
    mapping = request.backend_mapping
    config_module = _make_gui_config(mapping)
    options = dict(mapping["LOCOOptions"])
    rm_cfg = config_module.RMConfig(**mapping["RMConfig"])
    measured = _load_measurements(request.measurements, request.measurement_options)
    indices = _derive_indices(ring, measured)
    indices = _apply_machine_element_selections(
        indices, mapping.get("MachineElements", {}), rm_cfg
    )

    bad = _load_bad_bpm_positions(request.measurements)
    if bad is None and mapping.get("BadBPMPositions"):
        bad = np.asarray(mapping["BadBPMPositions"], dtype=int)
    if bad is not None:
        measured, indices = _apply_bad_bpm_positions(measured, indices, bad, remove_bad_bpms)
    excluded = mapping.get("ExcludedCorrectorPositions", {})
    if excluded.get("horizontal") or excluded.get("vertical"):
        measured, indices, rm_cfg.dkick = _apply_corrector_exclusions(
            measured, indices, excluded.get("horizontal", []),
            excluded.get("vertical", []), rm_cfg.dkick,
        )
    # The comparison evaluates the reference response only. Fitted magnet
    # groups are deliberately irrelevant here; BPM/corrector/cavity ordinals
    # remain strictly validated by _validate_indices.
    _validate_indices(ring, indices, ())

    expected = (indices["nHBPM"] + indices["nVBPM"], indices["nHorCOR"] + indices["nVerCOR"])
    if tuple(measured["orm"].shape) != expected:
        raise ValueError(
            f"Measured ORM shape {measured['orm'].shape} does not match the current ordered "
            f"FIT selection and exclusions; expected {expected}."
        )
    hstep, vstep = rm_cfg.dkick if isinstance(rm_cfg.dkick, (list, tuple)) else (rm_cfg.dkick, rm_cfg.dkick)
    hstep = _corrector_steps(hstep, indices["nHorCOR"], "horizontal")
    vstep = _corrector_steps(vstep, indices["nVerCOR"], "vertical")
    include_rf = bool(measured.get("dispersion_supplied", False))
    if cancelled():
        raise RuntimeError("Reference-model comparison cancelled.")
    # Mirror the model-response configuration constructed by pyloco() at the
    # start of each iteration.  In particular, do not feed unrelated GUI
    # fields into response_matrix: doing so can make this diagnostic differ
    # from model_orm_initial.h5 even though both start from the same lattice.
    response_config = config_module.RMConfig(
        dkick=[hstep, vstep],
        bpm_ords=indices["used_bpms_ords"],
        cm_ords=indices["used_cor_ords"],
        cav_ords=indices["CAVords"],
        HCMCoupling=np.zeros(indices["nHorCOR"], dtype=float),
        VCMCoupling=np.zeros(indices["nVerCOR"], dtype=float),
        rfStep=float(rm_cfg.rfStep),
        includeDispersion=include_rf,
        calculator=str(rm_cfg.calculator),
        bidirectional=bool(rm_cfg.bidirectional),
    )
    matrix = np.asarray(response_matrix(ring, config=response_config), dtype=float)
    if cancelled():
        raise RuntimeError("Reference-model comparison cancelled.")

    ncor = indices["nHorCOR"] + indices["nVerCOR"]
    model_orm = matrix[:, :ncor]
    model_rf = matrix[:, ncor] if include_rf else None
    measured_rf = (
        np.concatenate((measured["eta_x"], measured["eta_y"])) if include_rf else None
    )
    physical_measured = physical_model = None
    at_slip = None
    coefficient = None
    rf_frequency = float(config_module.fixed_parameters.Frequency)
    rf_step = float(rm_cfg.rfStep)
    if include_rf:
        if rf_step == 0:
            raise ValueError("The signed RF step is zero; physical dispersion cannot be calculated.")
        at_slip = float(ring.get_slip_factor())
        coefficient = at_slip * rf_frequency / rf_step
        physical_measured = measured_rf * coefficient
        physical_model = model_rf * coefficient

    lattice_path = Path(request.lattice_path).expanduser().resolve()
    provenance = {
        "lattice": str(lattice_path),
        "mat_variable": request.lattice_mat_variable or "not applicable",
        "lattice_sha256": sha256(lattice_path.read_bytes()).hexdigest(),
        "lattice_elements": len(ring),
        "measurements": dict(request.measurements),
        "calculator": str(rm_cfg.calculator),
        "bidirectional": bool(rm_cfg.bidirectional),
        "horizontal_kick_rad": np.asarray(hstep).tolist(),
        "vertical_kick_rad": np.asarray(vstep).tolist(),
        "rf_step_hz": rf_step,
        "rf_frequency_hz": rf_frequency,
        "at_slip_factor": at_slip,
        "physical_dispersion_coefficient": coefficient,
        "bpm_count": indices["nHBPM"],
        "horizontal_corrector_count": indices["nHorCOR"],
        "vertical_corrector_count": indices["nVerCOR"],
        "bad_bpm_positions": [] if bad is None else np.asarray(bad, dtype=int).tolist(),
        "excluded_horizontal_corrector_positions": list(excluded.get("horizontal", [])),
        "excluded_vertical_corrector_positions": list(excluded.get("vertical", [])),
        "residual_definition": "Measured − reference model",
        "rf_quantity": "signed RF-induced orbit difference [m]",
        "measured_rf_convention": (
            request.measurement_options.get("dispersion", {}).get("sign_convention")
            or request.measurement_options.get("dispersion", {}).get("measured_eta_definition")
            or "not recorded"
        ),
        "model_rf_convention": "response_matrix RF column for the configured signed rf_step_hz",
        "physical_definition": "D = RF orbit difference × eta_AT × f_RF / signed RF step",
    }
    return ReferenceComparisonResult(
        measured_orm=np.asarray(measured["orm"], dtype=float),
        model_orm=model_orm,
        measured_rf=measured_rf,
        model_rf=model_rf,
        physical_measured=physical_measured,
        physical_model=physical_model,
        n_bpms=indices["nHBPM"], n_hcors=indices["nHorCOR"], n_vcors=indices["nVerCOR"],
        cross_plane_excluded=bool(options.get("remove_coupling_", True)),
        statistics=_statistics(np.asarray(measured["orm"], dtype=float), model_orm,
                               indices["nHBPM"], indices["nHorCOR"]),
        rf_statistics=_rf_statistics(measured_rf, model_rf, indices["nHBPM"]),
        provenance=provenance,
    )
