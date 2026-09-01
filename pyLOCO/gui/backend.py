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
import pickle
import sys
import time
import traceback
import re
import copy
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
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
    measurement_options: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def results_root(self) -> Path:
        output = self.backend_mapping.get("Output", {}).get("directory")
        if output:
            return Path(output).expanduser().resolve()
        if self.project_path:
            return Path(self.project_path).expanduser().resolve().parent / "results"
        return Path.cwd() / "results"

    @classmethod
    def from_project(cls, project: ProjectMetadata) -> "LocoRunRequest":
        backend_mapping = project.loco_config.to_backend_mapping()
        output = backend_mapping.get("Output", {})
        if output.get("directory"):
            output["directory"] = str(project.resolve_path(output["directory"]))
        resume = backend_mapping.get("Resume", {})
        if resume.get("directory"):
            resume["directory"] = str(project.resolve_path(resume["directory"]))
        return cls(
            project_name=project.name,
            project_path=project.path,
            lattice_path=str(project.resolve_path(project.lattice.path)),
            measurements={
                key: str(project.resolve_path(dataset.path))
                for key, dataset in project.measurements.items()
            },
            measurement_options={key: dict(dataset.options) for key, dataset in project.measurements.items()},
            backend_mapping=backend_mapping,
        )


@dataclass(slots=True)
class LocoRunError:
    """Serializable backend exception details safe to emit across Qt threads."""

    message: str
    traceback: str
    cancelled: bool = False


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


def run_loco_request(
    request: LocoRunRequest,
    log_callback=None,
    cancel_callback=None,
    svd_selection_callback=None,
    progress_callback=None,
) -> LocoRunResult:
    """Execute the existing pyLOCO API for a GUI request.

    Cancellation is cooperative: the callback is forwarded to pyLOCO and checked
    at safe calculation checkpoints. Active Jacobian worker pools are terminated
    promptly without leaving a partially written result marked as completed.
    """

    start = time.monotonic()
    live_log = log_callback or (lambda message: None)
    log_lines: list[str] = []

    def log(message) -> None:
        text = str(message)
        log_lines.append(text)
        live_log(text)
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
        from pyLOCO.pyloco import pyloco, remove_bad_bpms, save_fit_dict

        ring = at.load_lattice(request.lattice_path)
        log(f"Loaded lattice: {request.lattice_path}")
        measured = _load_measurements(request.measurements, request.measurement_options)
        log("Loaded measurement files.")
        indices = _derive_indices(ring, measured)
        fit_cfg = config_module.FitInitConfig(**request.backend_mapping["FitInitConfig"])
        rm_cfg = config_module.RMConfig(**request.backend_mapping["RMConfig"])
        # Establish the user's complete machine selection before applying bad
        # BPM positions.  Reapplying the complete selection afterwards would
        # restore removed BPMs and make the ORM row count inconsistent.
        indices = _apply_machine_element_selections(
            indices, request.backend_mapping.get("MachineElements", {}), rm_cfg
        )
        bad_bpm_positions = _load_bad_bpm_positions(request.measurements)
        if bad_bpm_positions is None and request.backend_mapping.get("BadBPMPositions"):
            bad_bpm_positions = _as_bad_bpm_positions(request.backend_mapping["BadBPMPositions"])
        if bad_bpm_positions is not None:
            measured, indices = _apply_bad_bpm_positions(measured, indices, bad_bpm_positions, remove_bad_bpms)
            log(f"Applied Bad BPM list: removed {len(bad_bpm_positions)} BPM position(s).")
        exclusions = request.backend_mapping.get("ExcludedCorrectorPositions", {})
        if exclusions.get("horizontal") or exclusions.get("vertical"):
            measured, indices, rm_cfg.dkick = _apply_corrector_exclusions(
                measured, indices, exclusions.get("horizontal", []),
                exclusions.get("vertical", []), rm_cfg.dkick,
            )
            log("Applied corrector exclusions from selected-list positions.")
        log(
            "Using %d BPMs, %d horizontal correctors, %d vertical correctors."
            % (indices["nHBPM"], indices["nHorCOR"], indices["nVerCOR"])
        )

        mcf_cfg = request.backend_mapping.get("MomentumCompaction", {"source": "automatic"})
        try:
            import numpy as np
            mcf_value = np.asarray(config_module.get_mcf(ring), dtype=float)
            if mcf_value.size != 1 or not np.isfinite(mcf_value).all():
                raise ValueError("momentum compaction factor must be a finite scalar")
            log(f"Momentum compaction source: {mcf_cfg.get('source', 'automatic')}")
            log(f"Momentum compaction factor: {float(mcf_value.ravel()[0]):.6e}")
        except Exception as exc:
            raise ValueError(f"Unable to resolve momentum compaction factor: {exc}") from exc

        _validate_indices(ring, indices, fit_cfg.fit_list or ())
        constraint_cfg = _make_constraint_config(request.backend_mapping["ConstraintConfig"])
        options.setdefault("fit_list", fit_cfg.fit_list or ())
        # The GUI supplies its own Qt interactive-SVD dialog.  Matplotlib
        # windows must never be opened from this worker thread.
        _disable_worker_ui_options(options, log)

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
        resume_state = _load_resume_mapping(request.backend_mapping.get("Resume"), log)
        if resume_state is not None:
            kwargs.update(
                continue_from_previous=True,
                previous_ring=resume_state["ring"],
                previous_fit_dict=resume_state["fit_dict"],
                previous_fit_results=resume_state["fit_results"],
            )
        reference_ring = copy.deepcopy(resume_state["ring"] if resume_state is not None else ring)
        iteration_metrics: list[dict[str, Any]] = []
        calculator_trace: list[dict[str, Any]] = []
        jacobian_capture: dict[str, Any] = {}

        def persist_iteration(record: dict[str, Any]) -> None:
            diagnostics = _iteration_diagnostics(
                record,
                reference_ring=reference_ring,
                measured=measured,
                bpm_ords=indices["used_bpms_ords"],
                rf_step=float(rm_cfg.rfStep),
                rf_frequency=float(config_module.fixed_parameters.Frequency),
                momentum_compaction=float(mcf_value.ravel()[0]),
            )
            _save_iteration_snapshot(
                results_dir, record, diagnostics=diagnostics,
                reference_ring=reference_ring, measured=measured,
                include_dispersion=bool(options.get("includeDispersion", False)),
                bpm_ords=indices["used_bpms_ords"], rf_step=float(rm_cfg.rfStep),
                rf_frequency=float(config_module.fixed_parameters.Frequency),
                momentum_compaction=float(mcf_value.ravel()[0]), request=request,
            )
            if int(record.get("iteration", 0)) > 0:
                iteration_metrics.append(diagnostics)

        kwargs["initial_model_orm_callback"] = lambda orm: _save_initial_model_orm(results_dir, orm)
        kwargs["initial_state_callback"] = persist_iteration
        kwargs["iteration_metrics_callback"] = persist_iteration
        kwargs["calculator_trace_callback"] = calculator_trace.append
        kwargs["cancel_callback"] = cancelled
        if progress_callback is not None:
            kwargs["progress_callback"] = progress_callback
        if interactive_svd:
            if svd_selection_callback is None:
                raise RuntimeError(
                    "Interactive SVD selection requires a GUI selection callback."
                )
            kwargs["svd_selection_callback"] = svd_selection_callback
        if bool(options.get("save_jacobians", False)):
            kwargs["jacobian_callback"] = lambda matrix, iteration: jacobian_capture.update(
                matrix=matrix, iteration=int(iteration)
            )
        kwargs["output_dir"] = str(results_dir)
        (results_dir / "run_request.json").write_text(
            json.dumps(_jsonable(asdict(request)), indent=2), encoding="utf-8"
        )
        log("Starting pyLOCO backend execution...")
        stream = _ProgressStream(log)
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            result_tuple = pyloco(ring, **kwargs)
        stream.flush()
        if cancelled():
            raise RuntimeError("LOCO run cancelled by the user.")

        fit_results, fit_dict, final_ring, orm_model, c_bpms, chi2_history, delta_chi2_history, blocks = result_tuple
        elapsed = time.monotonic() - start
        initial_chi2 = _initial_chi2_from_log(log_lines)
        output_files = _save_outputs(
            results_dir, fit_results, fit_dict, final_ring, orm_model, c_bpms,
            chi2_history, delta_chi2_history, blocks, save_fit_dict,
            initial_chi2=initial_chi2, runtime_seconds=elapsed,
            resume_mapping=request.backend_mapping.get("Resume"),
            iteration_metrics=iteration_metrics,
            response_matrix_calculator=(
                "Tracking" if str(rm_cfg.calculator).strip().lower() == "numerical"
                else rm_cfg.calculator
            ),
            response_matrix_backend_calculator=rm_cfg.calculator,
            normal_quad_jacobian=options.get("quad_jacobian_calculator", "Numerical"),
            analytical_implementation=options.get("analytical_implementation", "vectorized"),
            analytical_dispersion_calculator=options.get(
                "analytical_dispersion_calculator"
            ),
            skew_analytical_implementation=options.get(
                "skew_analytical_implementation", "vectorized"
            ),
            skew_analytical_dispersion_calculator=options.get(
                "skew_analytical_dispersion_calculator"
            ),
            skew_analytical_dispersion_worker=options.get(
                "skew_analytical_dispersion_worker", "legacy_full_orm"
            ),
            calculator_trace=calculator_trace,
        )
        optics_path = _save_optics_results(
            results_dir,
            reference_ring=reference_ring,
            fitted_ring=final_ring,
            measured=measured,
            initial_orm_path=results_dir / "model_orm_initial.h5",
            fitted_orm=orm_model,
            include_dispersion=bool(options.get("includeDispersion", False)),
            reference_kind="resumed_fitted_lattice" if resume_state is not None else "run_input_lattice",
            bpm_ords=indices["used_bpms_ords"],
            rf_step=float(rm_cfg.rfStep),
            rf_frequency=float(config_module.fixed_parameters.Frequency),
            momentum_compaction=float(mcf_value.ravel()[0]),
        )
        if optics_path is not None:
            output_files.append(str(optics_path))
        manifest = _write_iteration_manifest(results_dir, run_status="completed")
        output_files.append(str(manifest))
        output_files.extend(str(path) for path in _save_jacobian(
            results_dir, jacobian_capture, blocks, request, indices
        ))
        log(f"LOCO run completed in {elapsed:.1f} s.")
        log_path = _save_backend_log(results_dir, log_lines)
        output_files.append(str(log_path))
        return LocoRunResult(str(results_dir), elapsed, [float(x) for x in chi2_history], output_files)
    except Exception:
        log(traceback.format_exc())
        if (results_dir / "iterations").is_dir():
            _write_iteration_manifest(results_dir, run_status="cancelled" if cancelled() else "failed")
        _save_backend_log(results_dir, log_lines)
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

    from dataclasses import dataclass, fields

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
    config_module.fixed_parameters = GUIFixedParameters(**mapping.get("FixedParameters", {}))
    mcf = mapping.get("MomentumCompaction", {"source": "automatic"})
    if mcf.get("source") == "user":
        value = float(mcf["value"])
        config_module.BACKEND = config_module.LOCOAPI(get_mcf=lambda ring, value=value: value)
    else:
        config_module.BACKEND = config_module.LOCOAPI()
    # The GUI mapping also carries newer pyloco call options which are not
    # fields of the legacy-compatible LOCOOptions dataclass.  Keep those in the
    # request mapping for _build_pyloco_kwargs, but do not pass them to this
    # constructor.
    option_fields = {item.name for item in fields(config_module.LOCOOptions)}
    legacy_options = {
        key: value for key, value in mapping.get("LOCOOptions", {}).items()
        if key in option_fields
    }
    config_module.loco_options = config_module.LOCOOptions(**legacy_options)
    return config_module


def _load_measurements(paths: dict[str, str], options: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    import numpy as np

    options = options or {}

    if "orm" not in paths:
        raise ValueError("An ORM measurement file is required.")
    orm_arrays = _load_array_mapping(paths["orm"])
    orm_options = options.get("orm", {})
    orm_name = str(orm_options.get("dataset") or "response_matrix")
    orm = np.asarray(_pick_array(orm_arrays, orm_name, "response_matrix", "orm"), dtype=float)
    if bool(orm_options.get("transpose", False)):
        orm = orm.T
    data = {"orm": orm * float(orm_options.get("scale", 1.0))}
    if data["orm"].ndim != 2 or data["orm"].shape[0] % 2:
        raise ValueError(f"ORM must be a two-dimensional array with an even row count; got {data['orm'].shape}.")
    n_bpms = data["orm"].shape[0] // 2
    if "dispersion" in paths:
        arrays = _load_array_mapping(paths["dispersion"])
        dispersion_options = options.get("dispersion", {})
        names = dispersion_options.get("datasets", {})
        data["eta_x"] = np.ravel(_pick_array(arrays, str(names.get("horizontal") or "measured_eta_x"), "eta_x", fallback_index=0)).astype(float) * float(dispersion_options.get("horizontal_scale", 1.0))
        data["eta_y"] = np.ravel(_pick_array(arrays, str(names.get("vertical") or "measured_eta_y"), "eta_y", fallback_index=1)).astype(float) * float(dispersion_options.get("vertical_scale", 1.0))
    else:
        data["eta_x"] = np.zeros(n_bpms)
        data["eta_y"] = np.zeros(n_bpms)
    data["dispersion_supplied"] = "dispersion" in paths
    if "bpm_noise" in paths:
        arrays = _load_array_mapping(paths["bpm_noise"])
        noise_options = options.get("bpm_noise", {})
        names = noise_options.get("datasets", {})
        data["noise_x"] = np.ravel(_pick_array(arrays, str(names.get("horizontal") or "Noise_BPMx"), "noise_x", fallback_index=0)).astype(float) * float(noise_options.get("horizontal_scale", 1.0))
        data["noise_y"] = np.ravel(_pick_array(arrays, str(names.get("vertical") or "Noise_BPMy"), "noise_y", fallback_index=1)).astype(float) * float(noise_options.get("vertical_scale", 1.0))
    else:
        data["noise_x"] = np.ones(n_bpms)
        data["noise_y"] = np.ones(n_bpms)
    for name in ("eta_x", "eta_y", "noise_x", "noise_y"):
        if data[name].size != n_bpms:
            raise ValueError(f"{name} length {data[name].size} does not match ORM BPM count {n_bpms}.")
        if not np.all(np.isfinite(data[name])):
            raise ValueError(f"{name} contains non-finite values.")
    return data


def _load_array_mapping(path: str | Path) -> dict[str, Any]:
    """Read supported scientific array containers without guessing physics."""
    import h5py
    import numpy as np

    source = Path(path).expanduser()
    if not source.exists():
        raise ValueError(f"Measurement file does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".npy":
        return {"array": np.load(source, allow_pickle=False)}
    if suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            return {key: np.array(archive[key]) for key in archive.files}
    if suffix in {".h5", ".hdf5"}:
        result = {}
        with h5py.File(source, "r") as handle:
            def collect_dataset(name, obj):
                # h5py stops traversal when a visitor returns a non-None value.
                # Store the dataset and return None explicitly so sibling
                # datasets (for example eta_x and eta_y) are also collected.
                if hasattr(obj, "shape"):
                    result[name] = np.array(obj)
                return None

            handle.visititems(collect_dataset)
        if not result:
            raise ValueError(f"Measurement file {source} contains no datasets.")
        return result
    if suffix == ".mat":
        if importlib.util.find_spec("scipy") is None:
            raise RuntimeError("SciPy is required to import MATLAB measurement files (.mat).")
        from scipy.io import loadmat
        return {key: value for key, value in loadmat(source).items() if not key.startswith("__")}
    raise ValueError(f"Unsupported measurement file type '{suffix}'. Use HDF5, MAT, NPY, or NPZ.")


def _pick_array(mapping: dict[str, Any], *names: str, fallback_index: int = 0):
    for name in names:
        if name in mapping:
            return mapping[name]
        matches = [value for key, value in mapping.items() if key.rsplit("/", 1)[-1] == name]
        if matches:
            return matches[0]
    values = list(mapping.values())
    if fallback_index >= len(values):
        raise ValueError(f"File does not contain an array for {names!r}.")
    return values[fallback_index]


def _load_bad_bpm_positions(paths: dict[str, str]):
    """Load an optional Bad BPM list from project measurements.

    Files may be .npy, .npz, .h5/.hdf5, or .mat and must contain a one-dimensional
    integer array of 0-based BPM positions. Preferred dataset/variable names are
    ``bad_bpm_positions`` or ``bad_bpms``; otherwise the first data array is used.
    """

    path = paths.get("bad_bpms")
    if not path:
        return None

    import h5py
    import numpy as np

    source = Path(path).expanduser()
    suffix = source.suffix.lower()
    if suffix == ".npy":
        values = np.load(source, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            key = _first_named_key(archive.keys(), "bad_bpm_positions", "bad_bpms")
            values = np.array(archive[key])
    elif suffix in {".h5", ".hdf5"}:
        with h5py.File(source, "r") as handle:
            dataset = _first_hdf5_dataset(handle, "bad_bpm_positions", "bad_bpms")
            values = np.array(dataset)
    elif suffix == ".mat":
        if importlib.util.find_spec("scipy") is None:
            raise RuntimeError("SciPy is required to import MATLAB Bad BPM list files (.mat).")
        from scipy.io import loadmat

        mat = {key: value for key, value in loadmat(source).items() if not key.startswith("__")}
        key = _first_named_key(mat.keys(), "bad_bpm_positions", "bad_bpms")
        values = mat[key]
    else:
        raise ValueError(f"Unsupported Bad BPM list file type '{suffix}'.")
    return _as_bad_bpm_positions(values)


def _first_named_key(keys, *preferred: str) -> str:
    keys = list(keys)
    for name in preferred:
        if name in keys:
            return name
    if not keys:
        raise ValueError("Bad BPM list file contains no arrays/datasets.")
    return keys[0]


def _first_hdf5_dataset(handle, *preferred: str):
    for name in preferred:
        if name in handle:
            return handle[name]
    datasets = []
    handle.visititems(lambda _name, obj: datasets.append(obj) if hasattr(obj, "shape") else None)
    if not datasets:
        raise ValueError(f"Bad BPM list file {handle.filename} contains no datasets.")
    return datasets[0]


def _as_bad_bpm_positions(values):
    import numpy as np

    array = np.asarray(values)
    array = np.squeeze(array)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"Bad BPM list must be one-dimensional; got shape {array.shape}.")
    if not np.issubdtype(array.dtype, np.integer):
        if np.issubdtype(array.dtype, np.floating) and np.all(np.isfinite(array)) and np.all(array == np.floor(array)):
            array = array.astype(np.int64)
        else:
            raise ValueError("Bad BPM list must contain integer BPM positions.")
    array = array.astype(np.int64, copy=False)
    if len(np.unique(array)) != len(array):
        raise ValueError("Bad BPM list must contain unique BPM positions; duplicate indices were found.")
    return array


def _apply_bad_bpm_positions(measured: dict[str, Any], indices: dict[str, Any], bad_bpm_positions, remove_bad_bpms):
    import numpy as np

    total_bpms = len(indices["used_bpms_ords"])
    if np.any(bad_bpm_positions < 0) or np.any(bad_bpm_positions >= total_bpms):
        raise ValueError(
            "Bad BPM list contains indices outside the valid 0-based BPM position range "
            f"[0, {total_bpms - 1}]."
        )
    updated_measured = dict(measured)
    updated_indices = dict(indices)
    updated_measured["noise_x"] = np.delete(updated_measured["noise_x"], bad_bpm_positions)
    updated_measured["noise_y"] = np.delete(updated_measured["noise_y"], bad_bpm_positions)
    updated_measured["eta_x"] = np.delete(updated_measured["eta_x"], bad_bpm_positions)
    updated_measured["eta_y"] = np.delete(updated_measured["eta_y"], bad_bpm_positions)
    updated_measured["orm"], _removed = remove_bad_bpms(
        updated_measured["orm"], bad_bpm_positions, total_bpms=total_bpms, axis=0, input_type="positions"
    )
    updated_indices["used_bpms_ords"] = np.delete(updated_indices["used_bpms_ords"], bad_bpm_positions)
    updated_indices["nHBPM"] = len(updated_indices["used_bpms_ords"])
    updated_indices["nVBPM"] = len(updated_indices["used_bpms_ords"])
    return updated_measured, updated_indices


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


def _apply_machine_element_selections(indices: dict[str, Any], selections: dict[str, Any], rm_cfg) -> dict[str, Any]:
    """Override auto-derived backend ordinals with explicit GUI element selections."""

    import numpy as np

    updated = dict(indices)
    bpm_ords = list(selections.get("bpm_ords") or rm_cfg.bpm_ords or [])
    hcor_ords = list(selections.get("horizontal_corrector_ords") or [])
    vcor_ords = list(selections.get("vertical_corrector_ords") or [])
    if not (hcor_ords or vcor_ords) and rm_cfg.cm_ords is not None:
        hcor_ords, vcor_ords = [list(v) for v in rm_cfg.cm_ords]
    cavity_ords = list(selections.get("cavity_ords") or rm_cfg.cav_ords or [])
    quad_ords = list(selections.get("normal_quadrupole_ords") or [])
    skew_ords = list(selections.get("skew_quadrupole_ords") or [])
    if bpm_ords:
        updated["used_bpms_ords"] = np.asarray(bpm_ords, dtype=int)
        updated["nHBPM"] = len(bpm_ords)
        updated["nVBPM"] = len(bpm_ords)
    if hcor_ords or vcor_ords:
        updated["used_cor_ords"] = [np.asarray(hcor_ords, dtype=int), np.asarray(vcor_ords, dtype=int)]
        updated["nHorCOR"] = len(hcor_ords)
        updated["nVerCOR"] = len(vcor_ords)
    if cavity_ords:
        updated["CAVords"] = np.asarray(cavity_ords, dtype=int)
    if quad_ords:
        updated["quads_ords"] = np.asarray(quad_ords, dtype=int)
        updated["quads_tilt_ind"] = np.asarray(quad_ords, dtype=int)
    if skew_ords:
        updated["skew_ords"] = np.asarray(skew_ords, dtype=int)
    return updated


def _validate_indices(ring, indices: dict[str, Any], fit_list) -> None:
    import numpy as np

    if indices["nHBPM"] <= 0:
        raise ValueError("No BPMs are selected or available in the lattice.")
    if indices["nHorCOR"] <= 0 or indices["nVerCOR"] <= 0:
        raise ValueError("Both horizontal and vertical correctors must be selected.")
    if "quads" in fit_list and _selection_size(indices.get("quads_ords")) == 0:
        raise ValueError("Quadrupole fitting was requested but no quadrupoles are selected.")
    if "skew_quads" in fit_list and _selection_size(indices.get("skew_ords")) == 0:
        raise ValueError("Skew-quadrupole fitting was requested but no skew quadrupoles are selected.")
    for label, values in (
        ("BPM", indices["used_bpms_ords"]),
        ("horizontal corrector", indices["used_cor_ords"][0]),
        ("vertical corrector", indices["used_cor_ords"][1]),
        ("quadrupole", indices.get("quads_ords")),
        ("skew quadrupole", indices.get("skew_ords")),
        ("cavity", indices.get("CAVords")),
    ):
        if values is None:
            continue
        array = np.asarray(values, dtype=int)
        if np.any(array < 0) or np.any(array >= len(ring)):
            raise ValueError(f"Selected {label} ordinal is outside the lattice range 0..{len(ring)-1}.")


def _selection_size(values) -> int:
    return 0 if values is None else len(values)


def _apply_corrector_exclusions(measured, indices, horizontal_positions, vertical_positions, dkick):
    """Remove zero-based selected-list positions and matching ORM columns."""
    import numpy as np

    h = _as_bad_bpm_positions(horizontal_positions) if horizontal_positions else np.array([], dtype=int)
    v = _as_bad_bpm_positions(vertical_positions) if vertical_positions else np.array([], dtype=int)
    nh, nv = int(indices["nHorCOR"]), int(indices["nVerCOR"])
    if np.any(h >= nh) or np.any(v >= nv):
        raise ValueError("Corrector exclusion contains a position outside the selected corrector list.")
    updated_measured = dict(measured)
    updated_measured["orm"] = np.delete(updated_measured["orm"], np.concatenate((h, nh + v)), axis=1)
    updated_indices = dict(indices)
    updated_indices["used_cor_ords"] = [
        np.delete(np.asarray(indices["used_cor_ords"][0]), h),
        np.delete(np.asarray(indices["used_cor_ords"][1]), v),
    ]
    updated_indices["nHorCOR"], updated_indices["nVerCOR"] = nh - h.size, nv - v.size
    values = dkick if isinstance(dkick, (list, tuple)) else (dkick, dkick)
    steps = []
    for value, count, removed in ((values[0], nh, h), (values[1], nv, v)):
        array = np.asarray(value, dtype=float)
        steps.append(np.delete(array.ravel(), removed) if array.ndim and array.size == count else value)
    return updated_measured, updated_indices, tuple(steps)


def _build_pyloco_kwargs(*, ring, options, rm_cfg, fit_cfg, constraint_cfg, fixed_parameters, measured, indices):
    import numpy as np

    if str(getattr(rm_cfg, "calculator", "Linear")).strip().lower() == "tracking":
        raise ValueError("Tracking mode is not a backend calculator name; use the supported Numerical calculator.")
    if not getattr(rm_cfg, "bidirectional", True):
        raise ValueError("The iterative backend assumes bidirectional ±kick/±RF measurements; one-sided fitting is not scientifically supported.")

    sigma_w = np.concatenate((measured["noise_x"], measured["noise_y"]))[:, np.newaxis]
    cmstep = rm_cfg.dkick if isinstance(rm_cfg.dkick, (list, tuple)) else (rm_cfg.dkick, rm_cfg.dkick)
    hstep = _corrector_steps(cmstep[0], indices["nHorCOR"], "horizontal")
    vstep = _corrector_steps(cmstep[1], indices["nVerCOR"], "vertical")
    expected_shape = (indices["nHBPM"] + indices["nVBPM"], indices["nHorCOR"] + indices["nVerCOR"])
    if measured["orm"].shape != expected_shape:
        raise ValueError(f"ORM shape {measured['orm'].shape} is incompatible with selected elements; expected {expected_shape}.")
    include_dispersion = options.get("includeDispersion", rm_cfg.includeDispersion)
    if include_dispersion and not measured.get("dispersion_supplied", True):
        raise ValueError("Dispersion fitting was requested but no dispersion measurement file was supplied.")
    measured_for_fit = _assemble_measured_response(measured, include_dispersion)
    return dict(
        algorithm=options.get("algorithm", "lm"), nIter=options.get("nIter", 1), **indices,
        orm_measured=measured_for_fit, weights=sigma_w, includeDispersion=include_dispersion,
        measured_eta_x=measured["eta_x"], measured_eta_y=measured["eta_y"],
        hor_dispersion_weight=options.get("hor_dispersion_weight", 1.0), ver_dispersion_weight=options.get("ver_dispersion_weight", 1.0),
        CMstep=[hstep, vstep], rfStep=rm_cfg.rfStep if rm_cfg.rfStep is not None else fixed_parameters.rfstep,
        Frequency=fixed_parameters.Frequency, fit_list=options.get("fit_list", ()),
        quad_individuals=fit_cfg.individuals,
        skew_individuals=options.get("skew_individuals", fit_cfg.individuals),
        tilt_individuals=options.get("tilt_individuals", fit_cfg.individuals),
        remove_coupling_=options.get("remove_coupling_", True), outlier_rejection=options.get("outlier_rejection", False),
        sigma_outlier=options.get("sigma_outlier", 10), apply_normalization=options.get("apply_normalization", False),
        normalization_mode=options.get("normalization_mode", "global"), svd_selection_method=options.get("svd_selection_method", "threshold"),
        svd_threshold=options.get("svd_threshold", 1e-7), cut_=options.get("cut_"), show_svd_plot=options.get("show_svd_plot", False),
        constraint_cfg=constraint_cfg, nLMIter=options.get("nLMIter", 10), Starting_Lambda=options.get("Starting_Lambda", 1e-3),
        max_lm_lambda=options.get("max_lm_lambda", 15), scaled=options.get("scaled", True), plot_fit_parameters=options.get("plot_fit_parameters", False),
        auto_correct_delta=options.get("auto_correct_delta", True), fixedpathlength=rm_cfg.fixedpathlength, fit_cfg=fit_cfg,
        calculate_delta_chi2=options.get("calculate_delta_chi2", False),
        response_matrix_calculator=getattr(rm_cfg, "calculator", "Linear"),
        quad_jacobian_calculator=options.get("quad_jacobian_calculator", "Numerical"),
        skew_jacobian_calculator=options.get("skew_jacobian_calculator", "Numerical"),
        analytical_thick_quadrupole=options.get("analytical_thick_quadrupole", True),
        analytical_thick_steerers=options.get("analytical_thick_steerers", False),
        analytical_verbose=options.get("analytical_verbose", False),
        analytical_use_mp=options.get("analytical_use_mp", False),
        analytical_implementation=options.get("analytical_implementation", "vectorized"),
        analytical_dispersion_calculator=options.get(
            "analytical_dispersion_calculator"
        ),
        analytical_thick_skew=options.get("analytical_thick_skew", True),
        analytical_skew_thick_steerers=options.get("analytical_skew_thick_steerers", False),
        analytical_skew_verbose=options.get("analytical_skew_verbose", False),
        analytical_skew_use_mp=options.get("analytical_skew_use_mp", False),
        skew_analytical_implementation=options.get(
            "skew_analytical_implementation", "vectorized"
        ),
        skew_analytical_dispersion_calculator=options.get(
            "skew_analytical_dispersion_calculator"
        ),
        skew_analytical_dispersion_worker=options.get(
            "skew_analytical_dispersion_worker", "legacy_full_orm"
        ),
        save_jacobians=options.get("save_jacobians", False),
    )


def _assemble_measured_response(measured, include_dispersion: bool):
    """Return canonical ORM ordering with ``[eta_x, eta_y]`` as the final column."""
    import numpy as np

    orm = np.asarray(measured["orm"], dtype=float)
    if not include_dispersion:
        return orm
    eta = np.concatenate((np.ravel(measured["eta_x"]), np.ravel(measured["eta_y"])))
    if eta.size != orm.shape[0]:
        raise ValueError(f"Dispersion vector length {eta.size} does not match ORM row count {orm.shape[0]}.")
    return np.hstack((orm, eta[:, None]))


def _load_resume_mapping(mapping: dict[str, Any] | None, log) -> dict[str, Any] | None:
    """Load the exact artifacts consumed by pyloco's continuation arguments."""
    if not mapping or not mapping.get("enabled"):
        return None
    directory = Path(str(mapping.get("directory") or "")).expanduser().resolve()
    results = directory / "results" if (directory / "results").is_dir() else directory
    ring_path = results / str(mapping.get("ring_file") or "ring_pyloco.mat")
    fit_dict_path = results / str(mapping.get("fit_dict_file") or "fit_dict.pkl")
    fit_results_name = mapping.get("fit_results_file")
    missing = [str(path) for path in (ring_path, fit_dict_path) if not path.is_file()]
    if fit_results_name and not (results / str(fit_results_name)).is_file():
        missing.append(str(results / str(fit_results_name)))
    if missing:
        raise ValueError("Resume state is incomplete; missing: " + ", ".join(missing))
    import at
    import numpy as np
    ring = at.load_lattice(ring_path)
    with fit_dict_path.open("rb") as stream:
        fit_dict = pickle.load(stream)
    fit_results = None
    if fit_results_name:
        fit_results = np.load(results / str(fit_results_name), allow_pickle=True).tolist()
    if not isinstance(fit_dict, dict):
        raise ValueError(f"Previous fit dictionary is incompatible: {fit_dict_path}")
    log(f"Initialization: resumed from {results}")
    return {"ring": ring, "fit_dict": fit_dict, "fit_results": fit_results,
            "results_directory": results}


def _corrector_steps(value, expected: int, plane: str):
    import numpy as np
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(expected, float(array))
    array = np.ravel(array)
    if array.size != expected:
        raise ValueError(f"{plane.title()} corrector-step length {array.size} does not match selected corrector count {expected}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{plane.title()} corrector steps must be finite.")
    return array


def _make_constraint_config(data: dict[str, Any]):
    return type("ConstraintConfig", (), data)() if data.get("enable") else None


def _make_results_dir(request: LocoRunRequest) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    configured = request.backend_mapping.get("Output", {}).get("run_name")
    name = configured or request.project_name
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(name)).strip("_") or "pyloco"
    path = request.results_root / f"{safe}-{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_initial_model_orm(results_dir: Path, orm_model) -> Path:
    import h5py
    import numpy as np

    path = results_dir / "model_orm_initial.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("response_matrix", data=np.asarray(orm_model), compression="gzip")
        handle.attrs["description"] = "Initial model ORM used for the first chi-squared evaluation"
    return path


def _save_optics(results_dir, initial_lattice_path, final_ring, indices):
    """Compatibility helper used by GUI regression tests and external callers."""
    if not initial_lattice_path or not indices:
        return None
    try:
        import at
        import numpy as np
        initial_ring = at.load_lattice(initial_lattice_path)
        bpm_ords = np.asarray(indices["used_bpms_ords"], dtype=int)
        _, initial_ringdata, initial_data = at.get_optics(initial_ring, refpts=bpm_ords)
        _, fitted_ringdata, fitted_data = at.get_optics(final_ring, refpts=bpm_ords)
        path = Path(results_dir) / "optics_results.npz"
        np.savez_compressed(
            path, bpm_ordinals=bpm_ords, s_position=np.asarray(initial_data.s_pos),
            beta_initial=np.asarray(initial_data.beta), beta_fitted=np.asarray(fitted_data.beta),
            tune_initial=np.asarray(initial_ringdata.tune), tune_fitted=np.asarray(fitted_ringdata.tune),
            definition=np.asarray("beta_beating=(beta_fitted-beta_initial)/beta_initial"),
        )
        return path
    except Exception:
        return None


def _save_jacobian(results_dir, capture, blocks, request, indices):
    """Persist the final combined Jacobian produced by the current backend.

    This complements the backend's optional per-block ``save_jacobians``
    artifacts; it does not recompute or replace them.
    """
    mapping = request.backend_mapping if request is not None else {}
    options = mapping.get("LOCOOptions", {})
    if not bool(options.get("save_jacobians", False)):
        return []
    if not capture or capture.get("matrix") is None:
        return []
    import h5py
    import numpy as np

    results_dir = Path(results_dir)
    matrix = np.asarray(capture["matrix"], dtype=float)
    artifact = results_dir / "jacobian.h5"
    with h5py.File(artifact, "w") as handle:
        handle.create_dataset("matrix", data=matrix, chunks=True, compression="gzip", shuffle=True)
    rm = mapping.get("RMConfig", {})
    metadata = {
        "shape": list(matrix.shape),
        "iteration": capture.get("iteration"),
        "storage": "HDF5 float64 dataset with lossless gzip compression",
        "parameter_blocks": _jsonable(blocks),
        "fitted_parameter_order": list((blocks or {}).keys()),
        "bpm_ordinals": _jsonable(indices.get("used_bpms_ords", []) if indices else []),
        "horizontal_corrector_ordinals": _jsonable(indices.get("used_cor_ords", [[], []])[0] if indices else []),
        "vertical_corrector_ordinals": _jsonable(indices.get("used_cor_ords", [[], []])[1] if indices else []),
        "dispersion_included": bool(options.get("includeDispersion", rm.get("includeDispersion", False))),
        "response_calculator": rm.get("calculator"),
        "normal_quad_jacobian": options.get("quad_jacobian_calculator", "Numerical"),
        "skew_quad_jacobian": options.get("skew_jacobian_calculator", "Numerical"),
        "perturbations": {
            "normal_quadrupole": mapping.get("FixedParameters", {}).get("dk"),
            "skew_quadrupole": mapping.get("FixedParameters", {}).get("delta_skew"),
            "quadrupole_tilt": mapping.get("FixedParameters", {}).get("delta_q_tilt"),
        },
        "row_order": "Fortran-flattened response matrix: BPM rows [horizontal, vertical] within each H/V-corrector column; optional dispersion is the final column.",
    }
    metadata_path = results_dir / "jacobian_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return [artifact, metadata_path]


def _iteration_diagnostics(
    record: dict[str, Any], *, reference_ring, measured: dict[str, Any],
    bpm_ords, rf_step: float, rf_frequency: float, momentum_compaction: float,
) -> dict[str, Any]:
    """Convert an in-process iteration snapshot to compact JSON metrics."""
    import numpy as np

    result = {key: value for key, value in record.items()
              if key not in {"ring", "orm_model", "fit_parameters", "blocks"}}
    ring = record.get("ring")

    def stats(values, *, scale=1.0):
        finite = np.asarray(values, dtype=float).ravel()
        finite = finite[np.isfinite(finite)] * scale
        if not finite.size:
            return {"rms": None, "mean": None, "max_abs": None}
        return {
            "rms": float(np.sqrt(np.mean(finite ** 2))),
            "mean": float(np.mean(finite)),
            "max_abs": float(np.max(np.abs(finite))),
        }

    try:
        refpts = np.arange(len(reference_ring), dtype=np.uint32)
        reference_beta = np.asarray(reference_ring.get_optics(refpts=refpts)[2].beta, dtype=float)
        fitted_beta = np.asarray(ring.get_optics(refpts=refpts)[2].beta, dtype=float)
        result["beta_beating_percent"] = {
            plane: stats(np.divide(
                fitted_beta[:, column] - reference_beta[:, column],
                reference_beta[:, column],
                out=np.full(reference_beta.shape[0], np.nan),
                where=reference_beta[:, column] != 0,
            ), scale=100.0)
            for plane, column in (("x", 0), ("y", 1))
        }
    except Exception as exc:
        result["beta_beating_unavailable_reason"] = str(exc)

    try:
        ords = np.asarray(bpm_ords, dtype=np.uint32).ravel()
        eta_x, eta_y = measured.get("eta_x"), measured.get("eta_y")
        if not measured.get("dispersion_supplied", False) or eta_x is None or eta_y is None:
            raise ValueError("measured dispersion was not supplied")
        conversion = -momentum_compaction * rf_frequency / rf_step
        fitted_dispersion = np.asarray(ring.get_optics(refpts=ords)[2].dispersion, dtype=float)
        result["dispersion_residual_m"] = {
            "x": stats(np.asarray(eta_x, dtype=float) * conversion - fitted_dispersion[:, 0]),
            "y": stats(np.asarray(eta_y, dtype=float) * conversion - fitted_dispersion[:, 2]),
        }
    except Exception as exc:
        result["dispersion_unavailable_reason"] = str(exc)

    return _jsonable(result)


def _save_iteration_snapshot(
    results_dir: Path, record: dict[str, Any], *, diagnostics: dict[str, Any],
    reference_ring, measured: dict[str, Any], include_dispersion: bool, bpm_ords,
    rf_step: float, rf_frequency: float, momentum_compaction: float,
    request: LocoRunRequest,
) -> Path:
    """Atomically persist one accepted outer-iteration state.

    The solver callback is emitted only after the corrected ORM and chi-squared
    have been calculated.  A temporary directory prevents a failed write from
    advertising a partial iteration as completed.
    """
    import at
    import numpy as np

    number = int(record.get("iteration", 0))
    root = results_dir / "iterations"
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"iteration_{number:03d}"
    temporary = Path(tempfile.mkdtemp(prefix=f".iteration_{number:03d}-", dir=root))
    try:
        vector = np.asarray(record["fit_parameters"], dtype=float).ravel()
        orm = np.asarray(record["orm_model"], dtype=float)
        np.savez_compressed(
            temporary / "loco_results.npz",
            fit_results=vector[np.newaxis, :], orm_model=orm,
            chi2_history=np.asarray([record.get("chi2_after")], dtype=float),
        )
        at.save_lattice(record["ring"], str(temporary / "fitted_lattice.mat"))
        _save_optics_results(
            temporary, reference_ring=reference_ring, fitted_ring=record["ring"],
            measured=measured, initial_orm_path=results_dir / "model_orm_initial.h5",
            fitted_orm=orm, include_dispersion=include_dispersion,
            reference_kind="run_initial_iteration", bpm_ords=bpm_ords,
            rf_step=rf_step, rf_frequency=rf_frequency,
            momentum_compaction=momentum_compaction,
        )
        previous_vector = None
        if number > 0:
            previous = root / f"iteration_{number - 1:03d}" / "loco_results.npz"
            if previous.exists():
                with np.load(previous, allow_pickle=False) as archive:
                    previous_vector = np.asarray(archive["fit_results"], dtype=float)[-1]
        initial_vector = vector
        initial = root / "iteration_000" / "loco_results.npz"
        if initial.exists():
            with np.load(initial, allow_pickle=False) as archive:
                initial_vector = np.asarray(archive["fit_results"], dtype=float)[-1]
        metadata = {
            **diagnostics,
            "schema_version": 1,
            "iteration": number,
            "label": "Initial" if number == 0 else f"Iteration {number}",
            "completed": True,
            "parent_run": "../..",
            "solver": request.backend_mapping.get("LOCOOptions", {}).get("algorithm"),
            "response_matrix_calculator": request.backend_mapping.get("RMConfig", {}).get("calculator"),
            "normal_quad_jacobian": request.backend_mapping.get("LOCOOptions", {}).get("quad_jacobian_calculator"),
            "skew_quad_jacobian": request.backend_mapping.get("LOCOOptions", {}).get("skew_jacobian_calculator"),
            "fit_parameter_blocks": _jsonable(record.get("blocks", {})),
            "fit_parameter_order": list((record.get("blocks") or {}).keys()),
            "dispersion_included": include_dispersion,
            "coupling_blocks": [name for name in (record.get("blocks") or {}) if "coupling" in name],
            "cumulative_parameter_change": _jsonable(vector - initial_vector),
            "iteration_parameter_step": _jsonable(vector - previous_vector) if previous_vector is not None else _jsonable(np.zeros_like(vector)),
        }
        (temporary / "iteration.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if final.exists():
            shutil.rmtree(final)
        temporary.replace(final)
        _write_iteration_manifest(results_dir)
        return final
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_iteration_manifest(results_dir: Path, *, run_status: str = "running") -> Path:
    root = results_dir / "iterations"
    entries = []
    for directory in sorted(root.glob("iteration_[0-9][0-9][0-9]")):
        metadata_path = directory / "iteration.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if metadata.get("completed") is True:
            entries.append({"iteration": int(metadata["iteration"]), "directory": directory.name,
                            "label": metadata.get("label")})
    path = root / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "run_status": run_status,
                                "iterations": entries}, indent=2), encoding="utf-8")
    return path


def _save_optics_results(
    results_dir: Path, *, reference_ring, fitted_ring, measured: dict[str, Any],
    initial_orm_path: Path, fitted_orm, include_dispersion: bool, reference_kind: str,
    bpm_ords=None, rf_step: float | None = None, rf_frequency: float | None = None,
    momentum_compaction: float | None = None,
) -> Path | None:
    """Persist factual optics derived from the run's actual reference/final lattices.

    Failure to calculate optional optics must never invalidate an otherwise
    successful LOCO fit.  Old runs remain supported by the results loader.
    """
    import numpy as np

    arrays: dict[str, Any] = {
        "schema_version": np.asarray(2),
        "reference_kind": np.asarray(reference_kind),
        "dispersion_included": np.asarray(include_dispersion),
        "dispersion_in_fit": np.asarray(include_dispersion),
        "dispersion_diagnostic_available": np.asarray(False),
    }
    try:
        refpts = np.arange(len(reference_ring), dtype=np.uint32)
        reference_optics = reference_ring.get_optics(refpts=refpts)
        fitted_optics = fitted_ring.get_optics(refpts=refpts)
        ref_data = reference_optics[2]
        fit_data = fitted_optics[2]
        beta_ref = np.asarray(ref_data.beta, dtype=float)
        beta_fit = np.asarray(fit_data.beta, dtype=float)
        if beta_ref.shape == beta_fit.shape and beta_ref.ndim == 2 and beta_ref.shape[1] >= 2:
            arrays.update({
                "s": np.asarray(reference_ring.get_s_pos(refpts), dtype=float),
                "beta_x_reference": beta_ref[:, 0], "beta_y_reference": beta_ref[:, 1],
                "beta_x_fitted": beta_fit[:, 0], "beta_y_fitted": beta_fit[:, 1],
                "beta_beating_x": np.divide(beta_fit[:, 0] - beta_ref[:, 0], beta_ref[:, 0],
                                             out=np.full(beta_ref.shape[0], np.nan), where=beta_ref[:, 0] != 0),
                "beta_beating_y": np.divide(beta_fit[:, 1] - beta_ref[:, 1], beta_ref[:, 1],
                                             out=np.full(beta_ref.shape[0], np.nan), where=beta_ref[:, 1] != 0),
            })
            for prefix, optics in (("reference", reference_optics), ("fitted", fitted_optics)):
                ring_data = optics[1]
                tune = getattr(ring_data, "tune", None)
                if tune is not None:
                    arrays[f"tune_{prefix}"] = np.asarray(tune, dtype=float)
    except Exception:
        pass

    eta_x, eta_y = measured.get("eta_x"), measured.get("eta_y")
    try:
        ords = np.asarray(bpm_ords, dtype=np.uint32).ravel()
        valid_conversion = rf_step not in (None, 0) and rf_frequency is not None and momentum_compaction is not None
        if not measured.get("dispersion_supplied", False):
            raise ValueError("measured dispersion was not supplied")
        if eta_x is None or eta_y is None or len(eta_x) != len(ords) or len(eta_y) != len(ords):
            raise ValueError("measured dispersion and BPM mapping lengths differ")
        if not valid_conversion:
            raise ValueError("RF frequency, RF step, or momentum compaction is unavailable")
        reference_dispersion = np.asarray(reference_ring.get_optics(refpts=ords)[2].dispersion, dtype=float)
        fitted_dispersion = np.asarray(fitted_ring.get_optics(refpts=ords)[2].dispersion, dtype=float)
        if reference_dispersion.shape[0] != len(ords) or fitted_dispersion.shape[0] != len(ords):
            raise ValueError("lattice dispersion and BPM mapping lengths differ")
        conversion = -float(momentum_compaction) * float(rf_frequency) / float(rf_step)
        arrays.update({
            "dispersion_diagnostic_available": np.asarray(True),
            "dispersion_measurement_convention": np.asarray("rf_orbit_difference_converted_by_-alpha_c_f_rf_over_delta_f"),
            "dispersion_rf_step_hz": np.asarray(float(rf_step)),
            "dispersion_rf_frequency_hz": np.asarray(float(rf_frequency)),
            "dispersion_momentum_compaction": np.asarray(float(momentum_compaction)),
            "dispersion_bpm_ords": ords,
            "dispersion_s": np.asarray(reference_ring.get_s_pos(ords), dtype=float),
            "dispersion_x_measured": np.asarray(eta_x, dtype=float) * conversion,
            "dispersion_y_measured": np.asarray(eta_y, dtype=float) * conversion,
            "dispersion_x_initial": reference_dispersion[:, 0],
            "dispersion_y_initial": reference_dispersion[:, 2],
            "dispersion_x_fitted": fitted_dispersion[:, 0],
            "dispersion_y_fitted": fitted_dispersion[:, 2],
        })
    except Exception as exc:
        arrays["dispersion_unavailable_reason"] = np.asarray(str(exc))

    if len(arrays) <= 3:
        return None
    path = results_dir / "optics_results.npz"
    np.savez_compressed(path, **arrays)
    return path


def _save_outputs(results_dir, fit_results, fit_dict, final_ring, orm_model, c_bpms, chi2_history, delta_chi2_history, blocks, save_fit_dict, *, initial_chi2=None, runtime_seconds=None, resume_mapping=None, iteration_metrics=None, response_matrix_calculator=None, response_matrix_backend_calculator=None, normal_quad_jacobian=None, analytical_implementation="vectorized", analytical_dispersion_calculator=None, skew_analytical_implementation="vectorized", skew_analytical_dispersion_calculator=None, skew_analytical_dispersion_worker="legacy_full_orm", calculator_trace=None):
    import numpy as np

    files = []
    initial_orm = results_dir / "model_orm_initial.h5"
    if initial_orm.exists():
        files.append(str(initial_orm))
    npz = results_dir / "loco_results.npz"
    np.savez_compressed(npz, fit_results=np.asarray(fit_results, dtype=object), orm_model=orm_model, c_bpms=c_bpms, chi2_history=np.asarray(chi2_history), delta_chi2_history=np.asarray(delta_chi2_history, dtype=object))
    files.append(str(npz))
    fit_json = results_dir / "fit_dict.json"
    save_fit_dict(fit_dict, fit_json)
    files.append(str(fit_json))
    fit_pickle = results_dir / "fit_dict.pkl"
    with fit_pickle.open("wb") as stream:
        pickle.dump(fit_dict, stream)
    files.append(str(fit_pickle))
    fit_results_path = results_dir / "fit_results.npy"
    np.save(fit_results_path, np.asarray(fit_results, dtype=object), allow_pickle=True)
    files.append(str(fit_results_path))
    blocks_path = results_dir / "blocks.pkl"
    with blocks_path.open("wb") as stream:
        pickle.dump(blocks, stream)
    files.append(str(blocks_path))
    metrics_csv = results_dir / "iteration_metrics.csv"
    _save_iteration_metrics_csv(metrics_csv, iteration_metrics or [])
    files.append(str(metrics_csv))
    summary = results_dir / "summary.json"
    summary.write_text(json.dumps({
        "initial_chi2": initial_chi2,
        "chi2_history": _jsonable(chi2_history),
        "blocks": _jsonable(blocks),
        "runtime_seconds": runtime_seconds,
        "response_matrix_calculator": response_matrix_calculator,
        "response_matrix_backend_calculator": response_matrix_backend_calculator,
        "normal_quad_jacobian": normal_quad_jacobian,
        "analytical_implementation": analytical_implementation,
        "analytical_dispersion_calculator": (
            analytical_dispersion_calculator or response_matrix_calculator
        ),
        "skew_analytical_implementation": skew_analytical_implementation,
        "skew_analytical_dispersion_calculator": (
            skew_analytical_dispersion_calculator or response_matrix_calculator
        ),
        "skew_analytical_dispersion_worker": skew_analytical_dispersion_worker,
        "normal_quad_jacobian_orm_calculator": (
            response_matrix_calculator if normal_quad_jacobian == "Numerical" else None
        ),
        "calculator_trace": _jsonable(calculator_trace or []),
        "iteration_metrics": _jsonable(iteration_metrics or []),
        "timings": {
            "total_backend_seconds": runtime_seconds,
            "iterations": _jsonable([
                item.get("timings", {}) for item in (iteration_metrics or [])
            ]),
        },
        "initialization": "resumed" if resume_mapping and resume_mapping.get("enabled") else "current_model",
        "resumed_from": (resume_mapping or {}).get("directory"),
    }, indent=2), encoding="utf-8")
    files.append(str(summary))
    try:
        import at
        lattice = results_dir / "final_lattice.mat"
        at.save_lattice(final_ring, str(lattice))
        files.append(str(lattice))
        resume_lattice = results_dir / "ring_pyloco.mat"
        at.save_lattice(final_ring, str(resume_lattice))
        files.append(str(resume_lattice))
    except Exception:
        pass
    return files


def _save_iteration_metrics_csv(path: Path, records: list[dict[str, Any]]) -> None:
    import csv

    columns = (
        "iteration", "chi2_before", "chi2_after", "orm_rms_m", "h_orm_rms_m",
        "v_orm_rms_m", "beta_x_rms_percent", "beta_y_rms_percent",
        "dispersion_x_rms_mm", "dispersion_y_rms_mm", "model_orm_seconds",
        "jacobian_seconds", "trial_orm_seconds", "final_orm_seconds",
        "total_orm_seconds", "iteration_seconds", "cumulative_seconds",
    )

    def get(record, *keys):
        value = record
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        return value

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            dx = get(record, "dispersion_residual_m", "x", "rms")
            dy = get(record, "dispersion_residual_m", "y", "rms")
            writer.writerow({
                "iteration": record.get("iteration"),
                "chi2_before": record.get("chi2_before"),
                "chi2_after": record.get("chi2_after"),
                "orm_rms_m": get(record, "orm_residual", "rms"),
                "h_orm_rms_m": get(record, "horizontal_orm_residual", "rms"),
                "v_orm_rms_m": get(record, "vertical_orm_residual", "rms"),
                "beta_x_rms_percent": get(record, "beta_beating_percent", "x", "rms"),
                "beta_y_rms_percent": get(record, "beta_beating_percent", "y", "rms"),
                "dispersion_x_rms_mm": None if dx is None else 1000.0 * dx,
                "dispersion_y_rms_mm": None if dy is None else 1000.0 * dy,
                **{key: get(record, "timings", key) for key in columns if key.endswith("_seconds")},
            })


def _initial_chi2_from_log(lines: list[str]) -> float | None:
    """Extract the exact value already printed by pyloco; never recompute χ²."""
    pattern = re.compile(r"^\s*Initial Chi²:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$")
    for line in lines:
        match = pattern.search(line)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _save_backend_log(results_dir: Path, lines: list[str]) -> Path:
    path = results_dir / "backend.log"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


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
