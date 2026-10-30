"""Read measurement acquisition metadata without changing numeric data."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import datetime


IMPORT_HINTS = {
    "orm": "HDF5 datasets: response_matrix/orm (or first 2-D array); NumPy .npy/.npz and MAT arrays are supported. Acquisition attributes such as dkick_rad and bidirectional are reused when present.",
    "dispersion": "HDF5 datasets measured_eta_x/eta_x and measured_eta_y/eta_y, or NumPy/MAT vectors. RF-step attributes are reused when present.",
    "bpm_noise": "HDF5/NumPy/MAT vectors for horizontal and vertical BPM noise (noise_x/noise_y where named).",
    "bad_bpms": "A 1-D integer array of zero-based positions within the selected BPM list; .npy, .npz, HDF5 and MAT are supported.",
    "other": "HDF5, MAT, .npy or .npz data retained with the project for later use.",
}


def inspect_measurement_metadata(path: str | Path, role: str) -> dict[str, Any]:
    source = Path(path)
    metadata: dict[str, Any] = {"source": "measurement metadata"}
    if source.suffix.lower() not in {".h5", ".hdf5"}:
        return metadata
    import h5py
    import numpy as np

    with h5py.File(source, "r") as handle:
        attrs = dict(handle.attrs)
        metadata["available_datasets"] = []
        handle.visititems(lambda name, obj: metadata["available_datasets"].append(name) if isinstance(obj, h5py.Dataset) else None)
    available = metadata["available_datasets"]
    if role == "orm":
        for name in ("response_matrix", "orm"):
            if name in available:
                metadata["dataset"] = name
                break
    elif role == "dispersion":
        horizontal = next((name for name in ("measured_eta_x", "eta_x") if name in available), None)
        vertical = next((name for name in ("measured_eta_y", "eta_y") if name in available), None)
        if horizontal and vertical:
            metadata["datasets"] = {"horizontal": horizontal, "vertical": vertical}
    elif role == "bpm_noise":
        horizontal = next((name for name in ("Noise_BPMx", "noise_x") if name in available), None)
        vertical = next((name for name in ("Noise_BPMy", "noise_y") if name in available), None)
        if horizontal and vertical:
            metadata["datasets"] = {"horizontal": horizontal, "vertical": vertical}
    def scalar(*names):
        for name in names:
            if name in attrs:
                value = np.asarray(attrs[name]).ravel()
                if value.size == 1:
                    return value[0].item() if hasattr(value[0], "item") else value[0]
        return None
    if role == "orm":
        kick = scalar("dkick_rad", "kick_rad", "corrector_kick_rad")
        if kick is not None:
            metadata["dkick_h"] = metadata["dkick_v"] = float(kick)
        bidirectional = scalar("bidirectional")
        if bidirectional is not None:
            metadata["bidirectional"] = bool(bidirectional)
    elif role == "dispersion":
        step = scalar("rf_step_hz", "rf_signed_step_hz", "rf_bipolar_separation_hz", "rf_full_step", "delta_f_hz", "df_hz")
        if step is None and "dispersion_difference" in attrs:
            import re
            description = str(attrs["dispersion_difference"])
            match = re.search(r"minus(\d+(?:\.\d+)?)Hz\s*-\s*orbit_plus(\d+(?:\.\d+)?)Hz", description, re.I)
            if match:
                step = -(float(match.group(1)) + float(match.group(2)))
                metadata["bidirectional"] = True
        if step is not None:
            metadata["rf_step_hz"] = float(step)
        bidirectional = scalar("bidirectional")
        if bidirectional is not None:
            metadata["bidirectional"] = bool(bidirectional)
        if bool(metadata.get("bidirectional")):
            from pyLOCO.dispersion_convention import canonical_dispersion_convention
            convention = canonical_dispersion_convention(attrs)
            metadata["rf_step_hz"] = convention.rf_step_hz
            metadata["horizontal_scale"] = convention.array_scale
            metadata["vertical_scale"] = convention.array_scale
            metadata["source_rf_difference_sign_convention"] = convention.source_sign
            metadata["rf_difference_sign_convention"] = "positive_minus_negative"
    return metadata


def measurement_display_fields(path: str | Path, options: dict[str, Any] | None = None) -> dict[str, str]:
    """Return authoritative acquisition time and identity for the FIT import table."""
    source=Path(path); options=options or {}; timestamp=None; machine="—"
    if not source.exists():
        return {"date":"Not available","time":"Not available","machine_profile":str(options.get("machine_profile") or options.get("machine_identity") or "—"),"timestamp_source":"file unavailable","tooltip":f"Measurement file is not currently available\n{source}"}
    if source.suffix.lower() in {".h5",".hdf5"}:
        import h5py
        with h5py.File(source,"r") as handle:
            attrs=dict(handle.attrs)
            embedded={}
            if "metadata/json" in handle:
                import json
                raw=handle["metadata/json"][()]; embedded=json.loads(raw.decode() if isinstance(raw,bytes) else raw)
            for key in ("acquisition_timestamp","acquisition_timestamp_utc","timestamp_utc","created_utc","timestamp"):
                if key in attrs or key in embedded:timestamp=attrs.get(key,embedded.get(key)); break
            machine_value=attrs.get("machine_profile") or attrs.get("machine_identity") or attrs.get("profile") or embedded.get("machine_identity") or embedded.get("machine_profile")
            if machine_value is not None:machine=str(machine_value.decode() if isinstance(machine_value,bytes) else machine_value)
    if machine=="—":machine=str(options.get("machine_profile") or options.get("machine_identity") or "—")
    authority="HDF5 acquisition metadata"
    try:
        if isinstance(timestamp,bytes):timestamp=timestamp.decode()
        moment=datetime.fromisoformat(str(timestamp).replace("Z","+00:00")).astimezone() if timestamp is not None else None
    except (ValueError,TypeError):moment=None
    if moment is None:
        if not source.exists():
            return {"date":"Not available","time":"Not available","machine_profile":machine,"timestamp_source":"file unavailable","tooltip":f"Measurement file is not currently available\n{source}"}
        moment=datetime.fromtimestamp(source.stat().st_mtime); authority="file modification time (fallback)"
    suffix=" *" if authority.startswith("file") else ""
    return {"date":moment.strftime("%d %b %Y")+suffix,"time":moment.strftime("%H:%M:%S"),"machine_profile":machine,"timestamp_source":authority,"tooltip":f"Time source: {authority}\n{source}"}
