"""Thin GUI orchestration layer for running the existing pyLOCO backend.

This module intentionally contains no numerical LOCO algorithms.  It adapts the
serializable GUI project state into the public backend API, captures stdout for
GUI logging, and saves the returned objects in a project results directory.
"""

from __future__ import annotations

import contextlib
import io
import json
import importlib.util
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models.project import ProjectMetadata


@dataclass(slots=True)
class LocoRunRequest:
    """Serializable snapshot of GUI state required to execute LOCO."""

    project_name: str
    project_path: str
    lattice_path: str
    measurements: dict[str, str]
    backend_mapping: dict[str, Any]

    @property
    def results_root(self) -> Path:
        if self.project_path:
            return Path(self.project_path).expanduser().resolve().parent / "results"
        return Path.cwd() / "results"

    @classmethod
    def from_project(cls, project: ProjectMetadata) -> "LocoRunRequest":
        return cls(
            project_name=project.name,
            project_path=project.path,
            lattice_path=project.lattice.path,
            measurements={key: dataset.path for key, dataset in project.measurements.items()},
            backend_mapping=project.loco_config.to_backend_mapping(),
        )


@dataclass(slots=True)
class LocoRunError:
    """Serializable backend exception details safe to emit across Qt threads."""

    message: str
    traceback: str


@dataclass(slots=True)
class LocoRunResult:
    """Summary of a completed backend run and saved output locations."""

    results_dir: str
    elapsed_seconds: float
    chi2_history: list[float]
    output_files: list[str]


class _ProgressStream(io.TextIOBase):
    def __init__(self, callback):
        self._callback = callback
        self._buffer = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._callback(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._callback(self._buffer)
            self._buffer = ""


def run_loco_request(request: LocoRunRequest, log_callback=None, cancel_callback=None) -> LocoRunResult:
    """Execute the existing pyLOCO API for a GUI request.

    Cancellation is cooperative: pyLOCO currently has no cancellation hook, so the
    callback is checked before and after the backend call.
    """

    start = time.monotonic()
    log = log_callback or (lambda message: None)
    cancelled = cancel_callback or (lambda: False)
    results_dir = _make_results_dir(request)
    log(f"Results directory: {results_dir}")
    if cancelled():
        raise RuntimeError("LOCO run cancelled before backend execution started.")

    config_module = _make_gui_config(request.backend_mapping)
    options = dict(request.backend_mapping["LOCOOptions"])
    interactive_svd = options.get("svd_selection_method") == "interactive"
    if not interactive_svd:
        _configure_worker_matplotlib()
    try:
        import at
        from pyLOCO.pyloco import pyloco, save_fit_dict

        ring = at.load_lattice(request.lattice_path)
        log(f"Loaded lattice: {request.lattice_path}")
        measured = _load_measurements(request.measurements)
        log("Loaded measurement files.")
        indices = _derive_indices(ring, measured)
        log(
            "Using %d BPMs, %d horizontal correctors, %d vertical correctors."
            % (indices["nHBPM"], indices["nHorCOR"], indices["nVerCOR"])
        )

        fit_cfg = config_module.FitInitConfig(**request.backend_mapping["FitInitConfig"])
        rm_cfg = config_module.RMConfig(**request.backend_mapping["RMConfig"])
        constraint_cfg = _make_constraint_config(request.backend_mapping["ConstraintConfig"])
        options.setdefault("fit_list", fit_cfg.fit_list or ())
        _disable_worker_ui_options(options, log, preserve_svd_ui=interactive_svd)

        kwargs = _build_pyloco_kwargs(
            ring=ring,
            options=options,
            rm_cfg=rm_cfg,
            fit_cfg=fit_cfg,
            constraint_cfg=constraint_cfg,
            fixed_parameters=config_module.fixed_parameters,
            measured=measured,
            indices=indices,
        )
        (results_dir / "run_request.json").write_text(
            json.dumps(_jsonable(asdict(request)), indent=2), encoding="utf-8"
        )
        log("Starting pyLOCO backend execution...")
        stream = _ProgressStream(log)
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            result_tuple = pyloco(ring, **kwargs)
        stream.flush()
        if cancelled():
            log("Cancellation was requested; backend finished before it could be interrupted.")

        fit_results, fit_dict, final_ring, orm_model, c_bpms, chi2_history, delta_chi2_history, blocks = result_tuple
        output_files = _save_outputs(
            results_dir, fit_results, fit_dict, final_ring, orm_model, c_bpms, chi2_history, delta_chi2_history, blocks, save_fit_dict
        )
        elapsed = time.monotonic() - start
        log(f"LOCO run completed in {elapsed:.1f} s.")
        return LocoRunResult(str(results_dir), elapsed, [float(x) for x in chi2_history], output_files)
    except Exception:
        log(traceback.format_exc())
        raise
    finally:
        _close_worker_matplotlib_figures()


def _configure_worker_matplotlib() -> None:
    """Force a non-GUI Matplotlib backend before backend code imports pyplot.

    The LOCO backend can optionally create diagnostic Matplotlib figures.  When
    launched from the Qt GUI those calls happen in a QThread, so a native GUI
    backend such as macOSX/QtAgg could instantiate NSWindow/QWidget objects off
    the main thread.  Agg keeps worker-thread execution computation-only.
    """

    os.environ.setdefault("MPLBACKEND", "Agg")
    if importlib.util.find_spec("matplotlib") is None:
        return

    import matplotlib

    matplotlib.use("Agg", force=True)


def _disable_worker_ui_options(options: dict[str, Any], log, preserve_svd_ui: bool = False) -> None:
    """Disable backend options that would create interactive windows."""

    disabled = []
    keys = ["plot_fit_parameters"] if preserve_svd_ui else ["show_svd_plot", "plot_fit_parameters"]
    for key in keys:
        if options.get(key):
            options[key] = False
            disabled.append(key)
    if disabled:
        log("Disabled interactive backend UI options in worker thread: " + ", ".join(disabled))


def _close_worker_matplotlib_figures() -> None:
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        plt.close("all")


def _make_gui_config(mapping: dict[str, Any]):
    """Return the internal pyLOCO config module configured from GUI state."""

    from dataclasses import dataclass

    import pyLOCO.config as config_module

    @dataclass
    class GUIFixedParameters:
        Frequency: float = 499664399.4230182
        HarmNumber: int = 3840
        rfstep: float = -3000.0
        dk: Any = None
        delta_skew: float = 1e-3
        delta_q_tilt: float = 1e-6

    config_module.LOCOOptions = config_module.INTERNAL_LOCOOptions
    config_module.RMConfig = config_module.INTERNAL_RMConfig
    config_module.FitInitConfig = config_module.INTERNAL_FitInitConfig
    config_module.FixedParameters = config_module.INTERNAL_FixedParameters
    config_module.fixed_parameters = GUIFixedParameters()
    config_module.loco_options = config_module.LOCOOptions(**mapping.get("LOCOOptions", {}))
    return config_module


def _load_measurements(paths: dict[str, str]) -> dict[str, Any]:
    import h5py
    import numpy as np

    data = {}
    with h5py.File(paths["orm"], "r") as f:
        data["orm"] = np.array(_dataset(f, "response_matrix"))
    with h5py.File(paths["dispersion"], "r") as f:
        data["eta_x"] = np.array(_dataset(f, "measured_eta_x", "eta_x"))
        data["eta_y"] = np.array(_dataset(f, "measured_eta_y", "eta_y", fallback_index=1))
    with h5py.File(paths["bpm_noise"], "r") as f:
        data["noise_x"] = np.array(_dataset(f, "Noise_BPMx", "noise_x"))
        data["noise_y"] = np.array(_dataset(f, "Noise_BPMy", "noise_y", fallback_index=1))
    return data


def _dataset(handle, *names: str, fallback_index: int = 0):
    for name in names:
        if name in handle:
            return handle[name]
    keys = list(handle.keys())
    if not keys:
        raise ValueError(f"Measurement file {handle.filename} contains no datasets.")
    if fallback_index >= len(keys):
        raise ValueError(f"Measurement file {handle.filename} does not contain a dataset for {names!r}.")
    return handle[keys[fallback_index]]


def _derive_indices(ring, measured: dict[str, Any]) -> dict[str, Any]:
    import at
    import numpy as np

    n_bpms = measured["orm"].shape[0] // 2
    n_cors = measured["orm"].shape[1]
    bpm_ords = np.asarray(at.get_refpts(ring, at.elements.Monitor))[:n_bpms]
    hcors = np.asarray(at.get_refpts(ring, at.elements.Corrector))[: n_cors // 2]
    vcors = np.asarray(at.get_refpts(ring, at.elements.Corrector))[n_cors // 2 : n_cors]
    if len(vcors) < n_cors - len(hcors):
        vcors = hcors[: n_cors - len(hcors)]
    quads = np.asarray(at.get_refpts(ring, at.elements.Quadrupole))
    cavs = np.asarray(at.get_refpts(ring, at.elements.RFCavity))
    return {"used_bpms_ords": bpm_ords, "used_cor_ords": [hcors, vcors], "quads_ords": quads, "skew_ords": None, "CAVords": cavs, "quads_tilt_ind": quads, "nHBPM": len(bpm_ords), "nVBPM": len(bpm_ords), "nHorCOR": len(hcors), "nVerCOR": len(vcors)}


def _build_pyloco_kwargs(*, ring, options, rm_cfg, fit_cfg, constraint_cfg, fixed_parameters, measured, indices):
    import numpy as np

    sigma_w = np.concatenate((measured["noise_x"], measured["noise_y"]))[:, np.newaxis]
    cmstep = rm_cfg.dkick if isinstance(rm_cfg.dkick, (list, tuple)) else (rm_cfg.dkick, rm_cfg.dkick)
    return dict(
        algorithm=options.get("algorithm", "lm"), nIter=options.get("nIter", 1), **indices,
        orm_measured=measured["orm"], weights=sigma_w, includeDispersion=options.get("includeDispersion", rm_cfg.includeDispersion),
        measured_eta_x=measured["eta_x"], measured_eta_y=measured["eta_y"],
        hor_dispersion_weight=options.get("hor_dispersion_weight", 1.0), ver_dispersion_weight=options.get("ver_dispersion_weight", 1.0),
        CMstep=[np.full(indices["nHorCOR"], cmstep[0]), np.full(indices["nVerCOR"], cmstep[1])], rfStep=rm_cfg.rfStep or fixed_parameters.rfstep,
        Frequency=fixed_parameters.Frequency, fit_list=options.get("fit_list", ()), individuals=fit_cfg.individuals,
        remove_coupling_=options.get("remove_coupling_", True), outlier_rejection=options.get("outlier_rejection", False),
        sigma_outlier=options.get("sigma_outlier", 10), apply_normalization=options.get("apply_normalization", False),
        normalization_mode=options.get("normalization_mode", "global"), svd_selection_method=options.get("svd_selection_method", "threshold"),
        svd_threshold=options.get("svd_threshold", 1e-7), cut_=options.get("cut_"), show_svd_plot=options.get("show_svd_plot", False),
        constraint_cfg=constraint_cfg, nLMIter=options.get("nLMIter", 10), Starting_Lambda=options.get("Starting_Lambda", 1e-3),
        max_lm_lambda=options.get("max_lm_lambda", 15), scaled=options.get("scaled", True), plot_fit_parameters=options.get("plot_fit_parameters", False),
        auto_correct_delta=options.get("auto_correct_delta", True), fixedpathlength=rm_cfg.fixedpathlength, fit_cfg=fit_cfg,
    )


def _make_constraint_config(data: dict[str, Any]):
    return type("ConstraintConfig", (), data)() if data.get("enable") else None


def _make_results_dir(request: LocoRunRequest) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in request.project_name).strip("_") or "pyloco"
    path = request.results_root / f"{safe}-{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_outputs(results_dir, fit_results, fit_dict, final_ring, orm_model, c_bpms, chi2_history, delta_chi2_history, blocks, save_fit_dict):
    import numpy as np

    files = []
    npz = results_dir / "loco_results.npz"
    np.savez_compressed(npz, fit_results=np.asarray(fit_results, dtype=object), orm_model=orm_model, c_bpms=c_bpms, chi2_history=np.asarray(chi2_history), delta_chi2_history=np.asarray(delta_chi2_history, dtype=object))
    files.append(str(npz))
    fit_json = results_dir / "fit_dict.json"
    save_fit_dict(fit_dict, fit_json)
    files.append(str(fit_json))
    summary = results_dir / "summary.json"
    summary.write_text(json.dumps({"chi2_history": _jsonable(chi2_history), "blocks": _jsonable(blocks)}, indent=2), encoding="utf-8")
    files.append(str(summary))
    try:
        import at
        lattice = results_dir / "final_lattice.mat"
        at.save_lattice(final_ring, str(lattice))
        files.append(str(lattice))
    except Exception:
        pass
    return files


def _jsonable(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if isinstance(value, slice): return {"start": value.start, "stop": value.stop, "step": value.step}
    if isinstance(value, Path): return str(value)
    return value
