"""pyLOCO measurement HDF5 schema version 1.0.

Canonical root datasets remain readable by the existing pyLOCO importers.
Additional groups preserve raw acquisition states and diagnostics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from ._json import json_safe

SCHEMA_VERSION = "1.0"
FILE_TYPE = "pyloco.measurement"
MEASUREMENT_KINDS = frozenset({"orm", "bpm_noise", "dispersion"})


def _array(value: Any, name: str, *, ndim: int | None = None, finite: bool = True) -> np.ndarray:
    result = np.asarray(value)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; got {result.shape}")
    if finite and result.dtype.kind in "fc" and not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _vectors_match(values: Mapping[str, Any], length: int) -> None:
    for name, value in values.items():
        if _array(value, name, ndim=1).size != length:
            raise ValueError(f"{name} length does not match expected length {length}")


def _create_text_dataset(group: h5py.Group, name: str, values: Sequence[str]) -> None:
    group.create_dataset(name, data=np.asarray(values, dtype=h5py.string_dtype("utf-8")))


def _initialize(path: str | Path, kind: str, metadata: Mapping[str, Any] | None) -> h5py.File:
    if kind not in MEASUREMENT_KINDS:
        raise ValueError(f"Unsupported measurement kind: {kind}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(destination, "w")
    handle.attrs.update({
        "pyloco_file_type": FILE_TYPE,
        "schema_version": SCHEMA_VERSION,
        "measurement_kind": kind,
        "units_convention": "SI",
    })
    handle.create_dataset(
        "metadata/json",
        data=json.dumps(json_safe(dict(metadata or {}))),
        dtype=h5py.string_dtype("utf-8"),
    )
    return handle


def write_orm(
    path: str | Path,
    *,
    response_matrix: Any,
    bpm_names: Sequence[str],
    horizontal_corrector_names: Sequence[str],
    vertical_corrector_names: Sequence[str],
    requested_kick_h_rad: Any,
    requested_kick_v_rad: Any,
    actual_kick_h_rad: Any,
    actual_kick_v_rad: Any,
    orbit_plus_m: Any,
    orbit_minus_m: Any,
    scaled: bool = False,
    direction: str = "bipolar",
    original_setpoints_rad: Any | None = None,
    requested_state_a_rad: Any | None = None,
    requested_state_b_rad: Any | None = None,
    actual_state_a_rad: Any | None = None,
    actual_state_b_rad: Any | None = None,
    final_setpoints_rad: Any | None = None,
    orbit_std_plus_m: Any | None = None,
    orbit_std_minus_m: Any | None = None,
    timestamps_plus_s: Any | None = None,
    timestamps_minus_s: Any | None = None,
    restoration_status: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> Path:
    """Write an ORM measurement retaining raw bipolar orbit states.

    Raw orbit arrays use ``(n_correctors, 2*n_bpms)`` with response rows
    ordered horizontal BPMs then vertical BPMs and columns ordered H then V.
    """
    matrix = _array(response_matrix, "response_matrix", ndim=2).astype(float)
    n_bpm, n_h, n_v = len(bpm_names), len(horizontal_corrector_names), len(vertical_corrector_names)
    expected = (2 * n_bpm, n_h + n_v)
    if matrix.shape != expected:
        raise ValueError(f"response_matrix shape {matrix.shape} does not match {expected}")
    _vectors_match({
        "requested_kick_h_rad": requested_kick_h_rad,
        "actual_kick_h_rad": actual_kick_h_rad,
    }, n_h)
    _vectors_match({
        "requested_kick_v_rad": requested_kick_v_rad,
        "actual_kick_v_rad": actual_kick_v_rad,
    }, n_v)
    raw_shape = (n_h + n_v, 2 * n_bpm)
    plus = _array(orbit_plus_m, "orbit_plus_m").astype(float)
    minus = _array(orbit_minus_m, "orbit_minus_m").astype(float)
    if plus.shape != minus.shape or plus.ndim not in (2, 3) or plus.shape[0] != raw_shape[0] or plus.shape[-1] != raw_shape[1]:
        raise ValueError(f"raw orbit shapes must be {raw_shape} or (correctors, samples, {2*n_bpm})")
    mean_plus=plus if plus.ndim==2 else np.mean(plus,axis=1); mean_minus=minus if minus.ndim==2 else np.mean(minus,axis=1)
    std_plus=np.zeros_like(mean_plus) if orbit_std_plus_m is None and plus.ndim==2 else np.std(plus,axis=1,ddof=0) if orbit_std_plus_m is None else _array(orbit_std_plus_m,"orbit_std_plus_m",ndim=2)
    std_minus=np.zeros_like(mean_minus) if orbit_std_minus_m is None and minus.ndim==2 else np.std(minus,axis=1,ddof=0) if orbit_std_minus_m is None else _array(orbit_std_minus_m,"orbit_std_minus_m",ndim=2)
    if std_plus.shape!=raw_shape or std_minus.shape!=raw_shape: raise ValueError("ORM orbit standard-deviation shapes are inconsistent")
    statuses=tuple(restoration_status or ("not_verified",)*raw_shape[0])
    if len(statuses)!=raw_shape[0]: raise ValueError("restoration_status length does not match correctors")

    with _initialize(path, "orm", metadata) as handle:
        handle.create_dataset("response_matrix", data=matrix, compression="gzip")
        handle.attrs["row_order"] = "horizontal_bpms,vertical_bpms"
        handle.attrs["column_order"] = "horizontal_correctors,vertical_correctors"
        handle.attrs["response_matrix_unit"] = "m/rad" if scaled else "m"
        handle.attrs["scaled"] = bool(scaled)
        handle.attrs["direction"] = str(direction)
        handle.attrs["canonical_definition"] = "state_a orbit minus state_b orbit; divided by actual effective kick only when scaled"
        names = handle.create_group("names")
        _create_text_dataset(names, "bpms", bpm_names)
        _create_text_dataset(names, "horizontal_correctors", horizontal_corrector_names)
        _create_text_dataset(names, "vertical_correctors", vertical_corrector_names)
        kicks = handle.create_group("kicks")
        kicks.attrs["unit"] = "rad"
        kicks.create_dataset("horizontal/requested", data=requested_kick_h_rad)
        kicks.create_dataset("horizontal/actual", data=actual_kick_h_rad)
        kicks.create_dataset("vertical/requested", data=requested_kick_v_rad)
        kicks.create_dataset("vertical/actual", data=actual_kick_v_rad)
        raw = handle.create_group("raw")
        raw.attrs["response_row_order"] = "horizontal_bpms,vertical_bpms"
        raw.attrs["corrector_order"] = "horizontal_correctors,vertical_correctors"
        raw.create_dataset("orbit_plus_m", data=plus, compression="gzip")
        raw.create_dataset("orbit_minus_m", data=minus, compression="gzip")
        raw["state_a_orbits_m"] = raw["orbit_plus_m"]
        raw["state_b_orbits_m"] = raw["orbit_minus_m"]
        raw.attrs["state_a"] = "+kick" if direction in {"bipolar","positive"} else "-kick"
        raw.attrs["state_b"] = "-kick" if direction=="bipolar" else "reference"
        raw.create_dataset("mean_orbit_plus_m",data=mean_plus)
        raw.create_dataset("mean_orbit_minus_m",data=mean_minus)
        raw.create_dataset("std_orbit_plus_m",data=std_plus)
        raw.create_dataset("std_orbit_minus_m",data=std_minus)
        for name,value in (("timestamps_plus_s",timestamps_plus_s),("timestamps_minus_s",timestamps_minus_s)):
            if value is not None:
                array=_array(value,name,ndim=2)
                if plus.ndim!=3 or array.shape!=plus.shape[:2]: raise ValueError(f"{name} must match corrector/sample dimensions")
                raw.create_dataset(name,data=array)
        state = handle.create_group("setpoints")
        state.attrs["unit"]="rad"
        optional_vectors=(("original",original_setpoints_rad),("requested_state_a",requested_state_a_rad),("requested_state_b",requested_state_b_rad),("actual_state_a",actual_state_a_rad),("actual_state_b",actual_state_b_rad),("final",final_setpoints_rad))
        for name,value in optional_vectors:
            if value is not None:
                array=_array(value,name,ndim=1)
                if array.size!=raw_shape[0]: raise ValueError(f"{name} length does not match correctors")
                state.create_dataset(name,data=array)
        _create_text_dataset(state,"restoration_status",statuses)
        _write_diagnostics(handle, diagnostics)
    return Path(path)


def write_bpm_noise(
    path: str | Path,
    *,
    noise_x_m: Any,
    noise_y_m: Any,
    bpm_names: Sequence[str],
    raw_orbits_x_m: Any,
    raw_orbits_y_m: Any,
    metadata: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> Path:
    n_bpm = len(bpm_names)
    _vectors_match({"noise_x_m": noise_x_m, "noise_y_m": noise_y_m}, n_bpm)
    raw_x = _array(raw_orbits_x_m, "raw_orbits_x_m", ndim=2).astype(float)
    raw_y = _array(raw_orbits_y_m, "raw_orbits_y_m", ndim=2).astype(float)
    if raw_x.shape != raw_y.shape or raw_x.shape[1] != n_bpm:
        raise ValueError("raw BPM-noise orbit arrays must be (n_samples, n_bpms) with equal shapes")
    with _initialize(path, "bpm_noise", metadata) as handle:
        # Both legacy and descriptive aliases are intentionally provided.
        handle.create_dataset("Noise_BPMx", data=noise_x_m)
        handle.create_dataset("Noise_BPMy", data=noise_y_m)
        handle["noise_x"] = handle["Noise_BPMx"]
        handle["noise_y"] = handle["Noise_BPMy"]
        _create_text_dataset(handle.create_group("names"), "bpms", bpm_names)
        handle.create_dataset("raw/orbits_x_m", data=raw_x, compression="gzip")
        handle.create_dataset("raw/orbits_y_m", data=raw_y, compression="gzip")
        _write_diagnostics(handle, diagnostics)
    return Path(path)


def write_dispersion(
    path: str | Path,
    *,
    measured_eta_x: Any,
    measured_eta_y: Any,
    bpm_names: Sequence[str],
    rf_frequency_hz: Any,
    raw_orbits_x_m: Any,
    raw_orbits_y_m: Any,
    rf_setpoint_hz: Any | None = None,
    rf_readback_hz: Any | None = None,
    rf_step_hz: float,
    bidirectional: bool,
    metadata: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    state_labels: Sequence[str] | None = None,
    operator_confirmed: Any | None = None,
    sample_timestamps_s: Any | None = None,
    restoration_status: str = "not_verified",
) -> Path:
    """Write derived canonical dispersion plus every raw RF-state orbit."""
    n_bpm = len(bpm_names)
    _vectors_match({"measured_eta_x": measured_eta_x, "measured_eta_y": measured_eta_y}, n_bpm)
    rf = _array(rf_frequency_hz, "rf_frequency_hz", ndim=1).astype(float)
    raw_x = _array(raw_orbits_x_m, "raw_orbits_x_m").astype(float)
    raw_y = _array(raw_orbits_y_m, "raw_orbits_y_m").astype(float)
    if raw_x.shape != raw_y.shape or raw_x.ndim not in (2, 3):
        raise ValueError("raw RF-state orbit arrays must have equal 2D or 3D shapes")
    if raw_x.shape[0] != rf.size or raw_x.shape[-1] != n_bpm:
        raise ValueError("raw RF-state orbit arrays must be (states, bpms) or (states, samples, bpms)")
    mean_x = raw_x if raw_x.ndim == 2 else np.mean(raw_x, axis=1)
    mean_y = raw_y if raw_y.ndim == 2 else np.mean(raw_y, axis=1)
    std_x = np.zeros_like(mean_x) if raw_x.ndim == 2 else np.std(raw_x, axis=1, ddof=0)
    std_y = np.zeros_like(mean_y) if raw_y.ndim == 2 else np.std(raw_y, axis=1, ddof=0)
    setpoint = rf if rf_setpoint_hz is None else _array(rf_setpoint_hz, "rf_setpoint_hz", ndim=1)
    readback = np.full(rf.shape, np.nan) if rf_readback_hz is None else _array(
        rf_readback_hz, "rf_readback_hz", ndim=1, finite=False
    )
    if setpoint.size != rf.size or readback.size != rf.size:
        raise ValueError("RF setpoint/readback lengths must match RF states")
    if float(rf_step_hz) == 0.0:
        raise ValueError("rf_step_hz may not be zero")
    labels = tuple(state_labels or (f"state_{index}" for index in range(rf.size)))
    if len(labels) != rf.size:
        raise ValueError("state_labels length must match RF states")
    confirmed = np.ones(rf.size, dtype=bool) if operator_confirmed is None else _array(
        operator_confirmed, "operator_confirmed", ndim=1
    ).astype(bool)
    if confirmed.size != rf.size:
        raise ValueError("operator_confirmed length must match RF states")
    timestamps = None
    if sample_timestamps_s is not None:
        timestamps = _array(sample_timestamps_s, "sample_timestamps_s", ndim=2)
        if raw_x.ndim != 3 or timestamps.shape != raw_x.shape[:2]:
            raise ValueError("sample timestamps must match (states, samples)")
    with _initialize(path, "dispersion", metadata) as handle:
        handle.create_dataset("measured_eta_x", data=measured_eta_x)
        handle.create_dataset("measured_eta_y", data=measured_eta_y)
        handle.attrs["rf_step_hz"] = float(rf_step_hz)
        handle.attrs["bidirectional"] = bool(bidirectional)
        handle.attrs["restoration_status"] = str(restoration_status)
        handle.attrs["measured_eta_definition"] = str((metadata or {}).get(
            "canonical_measured_eta_definition", "orbit difference for rf_step_hz"
        ))
        handle.attrs["measured_eta_unit"] = "m"
        if metadata and metadata.get("rf_difference_sign_convention") is not None:
            handle.attrs["rf_difference_sign_convention"] = str(metadata["rf_difference_sign_convention"])
        if metadata and metadata.get("momentum_compaction_factor") is not None:
            handle.attrs["momentum_compaction_factor"] = float(metadata["momentum_compaction_factor"])
            handle.attrs["relativistic_correction_inverse_gamma_squared"] = float(metadata["relativistic_correction_inverse_gamma_squared"])
            handle.attrs["at_slip_factor"] = float(metadata["at_slip_factor"])
            handle.attrs["slip_factor_eta"] = float(metadata["slip_factor_eta"])
            handle.attrs["slip_factor_convention"] = str(metadata.get("slip_factor_convention", ""))
            handle.attrs["momentum_relation"] = str(metadata.get("momentum_relation", ""))
            handle.attrs["delta_positive"] = float(metadata["delta_positive"])
            handle.attrs["delta_negative"] = float(metadata["delta_negative"])
            handle.attrs["delta_span"] = float(metadata["delta_span"])
            handle.attrs["physical_dispersion_definition"] = str(metadata.get("physical_dispersion_definition", ""))
        if metadata and metadata.get("rf_restoration_difference_hz") is not None:
            handle.attrs["rf_restoration_difference_hz"] = float(metadata["rf_restoration_difference_hz"])
        if metadata:
            for name in ("rf_original_hz", "rf_positive_hz", "rf_negative_hz", "rf_restored_hz"):
                if metadata.get(name) is not None:
                    handle.attrs[name] = float(metadata[name])
            if metadata.get("rf_positive_hz") is not None and metadata.get("rf_negative_hz") is not None:
                handle.attrs["rf_bipolar_separation_hz"] = (
                    float(metadata["rf_positive_hz"]) - float(metadata["rf_negative_hz"])
                )
                handle.attrs["rf_signed_step_hz"] = (
                    float(metadata["rf_negative_hz"]) - float(metadata["rf_positive_hz"])
                )
            if metadata.get("delta_span_definition") is not None:
                handle.attrs["delta_span_definition"] = str(metadata["delta_span_definition"])
        if metadata and metadata.get("rf_restoration_is_measurement_state") is not None:
            handle.attrs["rf_restoration_is_measurement_state"] = bool(metadata["rf_restoration_is_measurement_state"])
            handle.attrs["verify_restored_orbit"] = bool(metadata.get("verify_restored_orbit", False))
            handle.attrs["dispersion_measurement_states"] = str(metadata.get("dispersion_measurement_states", ""))
            handle.attrs["reference_after_role"] = str(metadata.get("reference_after_role", ""))
        _create_text_dataset(handle.create_group("names"), "bpms", bpm_names)
        raw = handle.create_group("raw")
        raw.attrs["orbit_unit"] = "m"
        raw.attrs["rf_unit"] = "Hz"
        raw.create_dataset("rf_frequency_hz", data=rf)
        raw.create_dataset("rf_setpoint_hz", data=setpoint)
        raw.create_dataset("rf_readback_hz", data=readback)
        raw.create_dataset("orbits_x_m", data=raw_x, compression="gzip")
        raw.create_dataset("orbits_y_m", data=raw_y, compression="gzip")
        raw.create_dataset("mean_orbits_x_m", data=mean_x)
        raw.create_dataset("mean_orbits_y_m", data=mean_y)
        raw.create_dataset("std_orbits_x_m", data=std_x)
        raw.create_dataset("std_orbits_y_m", data=std_y)
        raw.create_dataset("operator_confirmed", data=confirmed)
        _create_text_dataset(raw, "state_labels", labels)
        states_group = raw.create_group("states")
        for index,label in enumerate(labels):
            state_group=states_group.create_group(str(label))
            state_group.attrs["rf_setpoint_hz"]=float(setpoint[index])
            state_group.attrs["rf_readback_hz"]=float(readback[index])
            state_group.attrs["operator_confirmed"]=bool(confirmed[index])
            state_group.create_dataset("orbits_x_m",data=raw_x[index],compression="gzip")
            state_group.create_dataset("orbits_y_m",data=raw_y[index],compression="gzip")
            state_group.create_dataset("mean_orbit_x_m",data=mean_x[index])
            state_group.create_dataset("mean_orbit_y_m",data=mean_y[index])
        if timestamps is not None:
            raw.create_dataset("sample_timestamps_s", data=timestamps)
            for index,label in enumerate(labels):states_group[str(label)].create_dataset("sample_timestamps_s",data=timestamps[index])
        derived = handle.create_group("derived")
        derived.attrs["response_unit"] = "m/Hz"
        derived.create_dataset("rf_normalized_response_x_m_per_hz", data=np.asarray(measured_eta_x) / float(rf_step_hz))
        derived.create_dataset("rf_normalized_response_y_m_per_hz", data=np.asarray(measured_eta_y) / float(rf_step_hz))
        _write_diagnostics(handle, diagnostics)
    return Path(path)


def _write_diagnostics(handle: h5py.File, diagnostics: Mapping[str, Any] | None) -> None:
    if not diagnostics:
        return
    group = handle.create_group("diagnostics")
    for name, value in diagnostics.items():
        array = np.asarray(value)
        if array.dtype.kind in "OUS":
            group.create_dataset(name, data=json.dumps(json_safe(value)), dtype=h5py.string_dtype("utf-8"))
        else:
            group.create_dataset(name, data=array)


def validate_measurement_file(path: str | Path) -> dict[str, Any]:
    """Validate schema identity, required datasets, dimensions, and ordering."""
    source = Path(path)
    errors: list[str] = []
    with h5py.File(source, "r") as handle:
        kind = str(handle.attrs.get("measurement_kind", ""))
        if handle.attrs.get("pyloco_file_type") != FILE_TYPE:
            errors.append("invalid or missing pyloco_file_type")
        if str(handle.attrs.get("schema_version", "")) != SCHEMA_VERSION:
            errors.append("unsupported schema_version")
        required = {
            "orm": ("response_matrix", "kicks/horizontal/requested", "kicks/horizontal/actual",
                    "kicks/vertical/requested", "kicks/vertical/actual", "raw/orbit_plus_m", "raw/orbit_minus_m"),
            "bpm_noise": ("Noise_BPMx", "Noise_BPMy", "raw/orbits_x_m", "raw/orbits_y_m"),
            "dispersion": ("measured_eta_x", "measured_eta_y", "raw/rf_frequency_hz",
                           "raw/rf_setpoint_hz", "raw/rf_readback_hz", "raw/orbits_x_m", "raw/orbits_y_m"),
        }.get(kind)
        if required is None:
            errors.append(f"unsupported measurement_kind {kind!r}")
        else:
            errors.extend(f"missing dataset {name}" for name in required if name not in handle)
        if kind == "orm" and "response_matrix" in handle and handle["response_matrix"].ndim != 2:
            errors.append("response_matrix must be two-dimensional")
        if kind == "orm" and "response_matrix" in handle:
            matrix=handle["response_matrix"]
            if not np.isfinite(matrix[:]).all(): errors.append("response_matrix contains non-finite values")
            if str(handle.attrs.get("row_order",""))!="horizontal_bpms,vertical_bpms": errors.append("invalid ORM row order")
            if str(handle.attrs.get("column_order",""))!="horizontal_correctors,vertical_correctors": errors.append("invalid ORM column order")
            if str(handle.attrs.get("response_matrix_unit","m")) not in {"m","m/rad"}: errors.append("invalid ORM response unit")
            if all(name in handle for name in ("names/bpms","names/horizontal_correctors","names/vertical_correctors")):
                nb=len(handle["names/bpms"]); nh=len(handle["names/horizontal_correctors"]); nv=len(handle["names/vertical_correctors"])
                if matrix.shape!=(2*nb,nh+nv): errors.append("ORM dimensions do not match stored names")
                for name,length in (("kicks/horizontal/requested",nh),("kicks/horizontal/actual",nh),("kicks/vertical/requested",nv),("kicks/vertical/actual",nv)):
                    if name in handle and (handle[name].shape!=(length,) or not np.isfinite(handle[name][:]).all()): errors.append(f"{name} has invalid length or values")
                if all(name in handle for name in ("raw/orbit_plus_m","raw/orbit_minus_m")):
                    plus=handle["raw/orbit_plus_m"]; minus=handle["raw/orbit_minus_m"]
                    valid_shape=plus.shape==minus.shape and plus.ndim in (2,3) and plus.shape[0]==nh+nv and plus.shape[-1]==2*nb
                    if not valid_shape: errors.append("raw ORM orbit dimensions are inconsistent")
                    elif not np.isfinite(plus[:]).all() or not np.isfinite(minus[:]).all(): errors.append("raw ORM orbits contain non-finite values")
        if kind == "dispersion" and "measured_eta_x" in handle and "measured_eta_y" in handle:
            if handle["measured_eta_x"].shape != handle["measured_eta_y"].shape:
                errors.append("measured_eta_x/y shapes differ")
            if all(name in handle for name in ("raw/rf_frequency_hz","raw/orbits_x_m","raw/orbits_y_m")):
                rf_states=handle["raw/rf_frequency_hz"].shape[0]; raw_x=handle["raw/orbits_x_m"]; raw_y=handle["raw/orbits_y_m"]
                if raw_x.shape != raw_y.shape or raw_x.ndim not in (2,3):
                    errors.append("raw dispersion orbit shapes must be equal 2D or 3D arrays")
                elif raw_x.shape[0] != rf_states or raw_x.shape[-1] != handle["measured_eta_x"].shape[0]:
                    errors.append("raw dispersion state/BPM dimensions are inconsistent")
                if "raw/sample_timestamps_s" in handle and (raw_x.ndim != 3 or handle["raw/sample_timestamps_s"].shape != raw_x.shape[:2]):
                    errors.append("dispersion sample timestamps do not match state/sample dimensions")
                if "raw/state_labels" in handle:
                    labels = [item.decode() if isinstance(item, bytes) else str(item)
                              for item in handle["raw/state_labels"][:]]
                    if "negative" in labels and "positive" in labels:
                        negative, positive = labels.index("negative"), labels.index("positive")
                        means_x = raw_x[:] if raw_x.ndim == 2 else np.mean(raw_x[:], axis=1)
                        means_y = raw_y[:] if raw_y.ndim == 2 else np.mean(raw_y[:], axis=1)
                        expected_x = means_x[negative] - means_x[positive]
                        expected_y = means_y[negative] - means_y[positive]
                        if not np.allclose(handle["measured_eta_x"][:], expected_x, rtol=1e-12, atol=1e-15):
                            errors.append("measured_eta_x is not the historical negative-minus-positive RF orbit difference")
                        if not np.allclose(handle["measured_eta_y"][:], expected_y, rtol=1e-12, atol=1e-15):
                            errors.append("measured_eta_y is not the historical negative-minus-positive RF orbit difference")
                        if str(handle.attrs.get("rf_difference_sign_convention", "")) != "negative_minus_positive":
                            errors.append("missing or invalid RF difference sign convention")
                        frequencies = handle["raw/rf_readback_hz"][:]
                        if np.isfinite(frequencies[[negative, positive]]).all():
                            signed_step = float(frequencies[negative] - frequencies[positive])
                            if not np.isclose(float(handle.attrs.get("rf_step_hz", np.nan)), signed_step,
                                              rtol=0.0, atol=1e-9):
                                errors.append("rf_step_hz does not equal the historical signed step f- minus f+")
    if errors:
        raise ValueError(f"Invalid pyLOCO measurement file {source}: " + "; ".join(errors))
    return {"path": str(source), "kind": kind, "schema_version": SCHEMA_VERSION}
