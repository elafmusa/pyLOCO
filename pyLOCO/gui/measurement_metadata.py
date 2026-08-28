"""Read measurement acquisition metadata without changing numeric data."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
        metadata["datasets"] = []
        handle.visititems(lambda name, obj: metadata["datasets"].append(name) if isinstance(obj, h5py.Dataset) else None)
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
        step = scalar("rf_step_hz", "delta_f_hz", "df_hz")
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
    return metadata
