"""Lazy, Qt-free access to artifacts produced by a GUI pyLOCO run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OrmPartitions:
    n_hbpm: int
    n_vbpm: int
    n_hcor: int
    n_vcor: int


@dataclass(frozen=True)
class ParameterBlock:
    key: str
    label: str
    values: Any
    baseline: Any
    unit: str

    @property
    def changes(self):
        return self.values - self.baseline if self.baseline is not None else None


class ResultsLoader:
    """Translate saved pyLOCO artifacts into stable view-facing properties.

    Metadata is loaded independently of the potentially large ORM arrays. ORM
    metrics use the full numerical matrices; decimation belongs only to views.
    Missing optional files return ``None`` rather than inventing values.
    """

    def __init__(self, result_dir: str | Path, *, runtime: float | None = None) -> None:
        self.result_dir = Path(result_dir).expanduser().resolve()
        self._runtime_override = runtime
        self._cache: dict[str, Any] = {}
        self._unavailable: dict[str, str] = {}

    def unavailable_reason(self, quantity: str) -> str | None:
        """Return a developer-useful reason after an unavailable property was attempted."""
        return self._unavailable.get(quantity)

    def _json(self, name: str) -> dict[str, Any]:
        key = f"json:{name}"
        if key not in self._cache:
            path = self.result_dir / name
            try:
                value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            except (OSError, ValueError, TypeError):
                value = {}
            self._cache[key] = value if isinstance(value, dict) else {}
        return self._cache[key]

    @property
    def summary(self) -> dict[str, Any]:
        return self._json("summary.json")

    @property
    def request(self) -> dict[str, Any]:
        return self._json("run_request.json")

    @property
    def options(self) -> dict[str, Any]:
        return self.request.get("backend_mapping", {}).get("LOCOOptions", {})

    @property
    def chi2_history(self) -> list[float]:
        values = self.summary.get("chi2_history")
        if values is None:
            values = self._npz_value("chi2_history")
        if values is None:
            return []
        try:
            return [float(value) for value in values]
        except (TypeError, ValueError):
            return []

    @property
    def initial_chi2(self) -> float | None:
        value = self.summary.get("initial_chi2")
        return _finite_float(value)

    @property
    def final_chi2(self) -> float | None:
        values = self.chi2_history
        return values[-1] if values else None

    @property
    def chi2_reduction_percent(self) -> float | None:
        before, after = self.initial_chi2, self.final_chi2
        if before is None or after is None or before == 0:
            return None
        return 100.0 * (before - after) / abs(before)

    @property
    def requested_iterations(self) -> int | None:
        value = self.options.get("nIter")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def completed_iterations(self) -> int:
        return len(self.chi2_history)

    @property
    def fit_method(self) -> str | None:
        value = self.options.get("algorithm")
        return str(value).upper() if value else None

    @property
    def regularization(self) -> float | None:
        return _finite_float(self.options.get("Starting_Lambda"))

    @property
    def runtime(self) -> float | None:
        return self._runtime_override if self._runtime_override is not None else _finite_float(self.summary.get("runtime_seconds"))

    @property
    def dispersion_included(self) -> bool:
        return bool(self.options.get("includeDispersion", False))

    @property
    def initialization(self) -> str:
        return str(self.summary.get("initialization") or "current_model")

    @property
    def resumed_from(self) -> str | None:
        value = self.summary.get("resumed_from")
        return str(value) if value else None

    @property
    def fitted_parameter_blocks(self) -> dict[str, slice]:
        result = {}
        for name, value in (self.summary.get("blocks") or {}).items():
            if isinstance(value, dict) and value.get("start") is not None and value.get("stop") is not None:
                result[str(name)] = slice(int(value["start"]), int(value["stop"]), value.get("step"))
        return result

    @property
    def fitted_parameter_count(self) -> int | None:
        blocks = self.fitted_parameter_blocks
        return max((block.stop for block in blocks.values()), default=None)

    @property
    def parameter_vector(self):
        if "parameter_vector" not in self._cache:
            import numpy as np
            raw = self._npz_value("fit_results")
            try:
                array = np.asarray(raw.tolist() if getattr(raw, "dtype", None) == object else raw, dtype=float)
                self._cache["parameter_vector"] = array[-1].ravel() if array.ndim > 1 else array.ravel()
            except (TypeError, ValueError, IndexError, AttributeError):
                self._cache["parameter_vector"] = None
                self._unavailable["parameter_vector"] = "No numeric final fit vector was persisted"
        return self._cache["parameter_vector"]

    @property
    def parameter_blocks(self) -> list[ParameterBlock]:
        """Return persisted fitted values split only by the saved backend slices."""
        vector = self.parameter_vector
        if vector is None:
            return []
        import numpy as np
        labels = {
            "hbpm_gain": ("Horizontal BPM gain", "dimensionless", 1.0),
            "vbpm_gain": ("Vertical BPM gain", "dimensionless", 1.0),
            "hbpm_coupling": ("Horizontal BPM coupling", "dimensionless", 0.0),
            "vbpm_coupling": ("Vertical BPM coupling", "dimensionless", 0.0),
            "hcor_cal": ("Horizontal corrector kick", "rad", None),
            "vcor_cal": ("Vertical corrector kick", "rad", None),
            "hcor_coupling": ("Horizontal corrector coupling", "dimensionless", 0.0),
            "vcor_coupling": ("Vertical corrector coupling", "dimensionless", 0.0),
            "HCMEnergyShift": ("Horizontal corrector energy shift", "backend value", 0.0),
            "VCMEnergyShift": ("Vertical corrector energy shift", "backend value", 0.0),
            "delta_rf": ("RF frequency shift", "Hz", 0.0),
            "quads": ("Normal quadrupole strength", "m⁻²", None),
            "skew_quads": ("Skew quadrupole strength", "m⁻²", None),
            "quads_tilt": ("Quadrupole tilt", "rad", 0.0),
        }
        result = []
        cmstep = self.request.get("backend_mapping", {}).get("FitInitConfig", {}).get("CMstep")
        for key, block in self.fitted_parameter_blocks.items():
            if block.stop > vector.size:
                continue
            label, unit, initial = labels.get(key, (key.replace("_", " ").title(), "backend value", None))
            values = np.asarray(vector[block], dtype=float)
            baseline = np.full(values.shape, initial) if initial is not None else None
            if key in {"quads", "skew_quads"}:
                candidate = self._lattice_parameter_baseline(key, values.size)
                if candidate is not None:
                    baseline = candidate
            if key in {"hcor_cal", "vcor_cal"} and isinstance(cmstep, (list, tuple)) and len(cmstep) >= 2:
                candidate = np.asarray(cmstep[0 if key == "hcor_cal" else 1], dtype=float).ravel()
                if candidate.size == values.size:
                    baseline = candidate
            result.append(ParameterBlock(key, label, values, baseline, unit))
        return result

    def _lattice_parameter_baseline(self, key: str, expected: int):
        """Read the selected magnet values from the run's initial lattice."""
        import numpy as np

        mapping = self.request.get("backend_mapping", {})
        elements = mapping.get("MachineElements", {})
        fit_init = mapping.get("FitInitConfig", {})
        if key == "quads":
            ordinals = elements.get("normal_quadrupole_ords") or []
            attribute = fit_init.get("quads_attr") or "PolynomB"
            attribute_index = int(fit_init.get("quads_attr_index", 1))
        else:
            ordinals = elements.get("skew_quadrupole_ords") or []
            attribute = fit_init.get("skew_attr") or "PolynomA"
            attribute_index = int(fit_init.get("skew_attr_index", 1))
        if len(ordinals) != expected:
            return None
        lattice_path = self.request.get("lattice_path")
        if not lattice_path:
            return None
        try:
            import at

            ring = at.load_lattice(self._resolve_reference(lattice_path))
            values = []
            for ordinal in ordinals:
                value = getattr(ring[int(ordinal)], attribute)
                array = np.asarray(value)
                values.append(float(array[attribute_index]) if array.ndim else float(array))
            result = np.asarray(values, dtype=float)
            return result if result.size == expected and np.isfinite(result).all() else None
        except Exception:
            return None

    @property
    def input_files(self) -> list[tuple[str, Path | None]]:
        values = []
        lattice = self.request.get("lattice_path") or self.request.get("machine", {}).get("lattice")
        if lattice:
            values.append(("Lattice", self._resolve_reference(lattice)))
        for role, value in (self.request.get("measurements") or {}).items():
            if value:
                values.append((role.replace("_", " ").title(), self._resolve_reference(value)))
        return values

    @property
    def dispersion_data(self) -> dict[str, Any] | None:
        """Return an independent physical-dispersion diagnostic when possible."""
        if "dispersion_data" in self._cache:
            return self._cache["dispersion_data"]
        persisted = self.optics_results
        available = bool(_scalar_value(persisted.get("dispersion_diagnostic_available"), False))
        if available:
            data = {}
            for plane in ("x", "y"):
                plane_data = {kind: persisted.get(f"dispersion_{plane}_{kind}") for kind in ("measured", "initial", "fitted")}
                if all(value is not None for value in plane_data.values()):
                    data[plane] = plane_data
            if data:
                data["axis"] = persisted.get("dispersion_s")
                data["axis_label"] = "Longitudinal position s [m]" if data["axis"] is not None else "BPM index in saved ordering"
                self._cache["dispersion_data"] = data
                return data
        self._cache["dispersion_data"] = self._legacy_dispersion_diagnostic()
        return self._cache["dispersion_data"]

    @property
    def dispersion_unavailable_reason(self) -> str | None:
        if self.dispersion_data is not None:
            return None
        persisted = self.optics_results
        value = persisted.get("dispersion_unavailable_reason")
        return str(_scalar_value(value, self._unavailable.get("dispersion_diagnostic", "Dispersion diagnostic data are unavailable.")))

    @property
    def dispersion_statistics(self) -> dict[str, dict[str, float | None]]:
        import numpy as np
        result = {}
        for plane, values in (self.dispersion_data or {}).items():
            if plane not in {"x", "y"}: continue
            measured = np.asarray(values["measured"], dtype=float)
            before = measured - np.asarray(values["initial"], dtype=float)
            after = measured - np.asarray(values["fitted"], dtype=float)
            def metrics(residual):
                finite = residual[np.isfinite(residual)]
                return {"rms": float(np.sqrt(np.mean(finite**2))), "mean": float(np.mean(finite)),
                        "min": float(np.min(finite)), "max": float(np.max(finite)),
                        "max_abs": float(np.max(np.abs(finite)))} if finite.size else {k: None for k in ("rms", "mean", "min", "max", "max_abs")}
            before_metrics, after_metrics = metrics(before), metrics(after)
            improvement = None if before_metrics["rms"] in (None, 0) else 100.0 * (1.0 - after_metrics["rms"] / before_metrics["rms"])
            result[plane] = {**{f"{k}_before": v for k, v in before_metrics.items()},
                             **{f"{k}_after": v for k, v in after_metrics.items()}, "improvement": improvement}
        return result

    def _legacy_dispersion_diagnostic(self):
        """Rebuild old-run diagnostics using recorded inputs and pyLOCO's RF convention."""
        import numpy as np
        measurements = self.request.get("measurements") or {}
        dispersion_file = measurements.get("dispersion")
        if not dispersion_file:
            self._unavailable["dispersion_diagnostic"] = "Measured dispersion is not available for this run."
            return None
        if self.initialization == "resumed":
            self._unavailable["dispersion_diagnostic"] = "The recorded reference lattice for this resumed run is unavailable."
            return None
        reference_path = self._resolve_reference(self.request.get("lattice_path", ""))
        fitted_path = self.result_dir / "final_lattice.mat"
        if reference_path is None:
            self._unavailable["dispersion_diagnostic"] = "Initial/reference lattice is unavailable."
            return None
        if not fitted_path.exists():
            self._unavailable["dispersion_diagnostic"] = "Final fitted lattice is unavailable."
            return None
        mapping = self.request.get("backend_mapping", {})
        machine = mapping.get("MachineElements", {})
        rm = mapping.get("RMConfig", {})
        bpm_ords = machine.get("bpm_ords") or rm.get("bpm_ords") or []
        if not bpm_ords:
            self._unavailable["dispersion_diagnostic"] = "BPM mapping required for dispersion comparison is unavailable."
            return None
        options = (self.request.get("measurement_options") or {}).get("dispersion", {})
        names = options.get("datasets", {})
        eta_x = self._load_vector_reference(dispersion_file, dataset_names=(str(names.get("horizontal") or "measured_eta_x"), "eta_x"))
        eta_y = self._load_vector_reference(dispersion_file, dataset_names=(str(names.get("vertical") or "measured_eta_y"), "eta_y"), fallback_index=1)
        if eta_x is None or eta_y is None:
            self._unavailable["dispersion_diagnostic"] = "Measured horizontal or vertical dispersion is unavailable."
            return None
        eta_x = eta_x * float(options.get("horizontal_scale", 1.0)); eta_y = eta_y * float(options.get("vertical_scale", 1.0))
        bad = list(mapping.get("BadBPMPositions") or [])
        if bad:
            eta_x, eta_y = np.delete(eta_x, bad), np.delete(eta_y, bad)
            bpm_ords = [value for index, value in enumerate(bpm_ords) if index not in set(bad)]
        if len(eta_x) != len(bpm_ords) or len(eta_y) != len(bpm_ords):
            self._unavailable["dispersion_diagnostic"] = "Measured dispersion and recorded BPM mapping have different lengths."
            return None
        rf_step = rm.get("rfStep")
        frequency = mapping.get("FixedParameters", {}).get("Frequency")
        if rf_step in (None, 0) or frequency is None:
            self._unavailable["dispersion_diagnostic"] = "RF frequency or RF step required for dispersion conversion is unavailable."
            return None
        try:
            import at
            from pyLOCO.config import get_mcf
            reference, fitted = at.load_lattice(reference_path), at.load_lattice(fitted_path)
            ords = np.asarray(bpm_ords, dtype=np.uint32)
            initial_dispersion = np.asarray(reference.get_optics(refpts=ords)[2].dispersion, dtype=float)
            fitted_dispersion = np.asarray(fitted.get_optics(refpts=ords)[2].dispersion, dtype=float)
            alpha_c = float(np.asarray(get_mcf(reference)).ravel()[0])
            conversion = -alpha_c * float(frequency) / float(rf_step)
            return {"x": {"measured": eta_x * conversion, "initial": initial_dispersion[:, 0], "fitted": fitted_dispersion[:, 0]},
                    "y": {"measured": eta_y * conversion, "initial": initial_dispersion[:, 2], "fitted": fitted_dispersion[:, 2]},
                    "axis": np.asarray(reference.get_s_pos(ords), dtype=float), "axis_label": "Longitudinal position s [m]"}
        except Exception as exc:
            self._unavailable["dispersion_diagnostic"] = f"Could not calculate lattice dispersion: {exc}"
            return None

    @property
    def optics_results(self) -> dict[str, Any]:
        if "optics_results" not in self._cache:
            import numpy as np
            path = self.result_dir / "optics_results.npz"
            try:
                with np.load(path, allow_pickle=False) as archive:
                    self._cache["optics_results"] = {key: np.array(archive[key]) for key in archive.files}
            except (OSError, ValueError):
                self._cache["optics_results"] = {}
        return self._cache["optics_results"]

    @property
    def beta_beating_data(self) -> dict[str, Any] | None:
        values = self.optics_results
        required = ("s", "beta_beating_x", "beta_beating_y")
        if all(key in values for key in required):
            return {key: values[key] for key in required} | {
                "reference_kind": str(values.get("reference_kind", "run_input_lattice")),
            }
        # Scientifically sufficient legacy case: a non-resumed run records its
        # exact input lattice and final_lattice.mat.  Never infer a resumed
        # reference if that prior lattice is not explicitly available.
        if self.initialization == "resumed":
            return None
        reference = self._resolve_reference(self.request.get("lattice_path", ""))
        fitted = self.result_dir / "final_lattice.mat"
        if reference is None or not fitted.exists():
            return None
        try:
            import at
            import numpy as np
            ref_ring, fit_ring = at.load_lattice(reference), at.load_lattice(fitted)
            if len(ref_ring) != len(fit_ring):
                return None
            refpts = np.arange(len(ref_ring), dtype=np.uint32)
            ref_beta = np.asarray(ref_ring.get_optics(refpts=refpts)[2].beta, dtype=float)
            fit_beta = np.asarray(fit_ring.get_optics(refpts=refpts)[2].beta, dtype=float)
            if ref_beta.shape != fit_beta.shape or ref_beta.ndim != 2 or ref_beta.shape[1] < 2:
                return None
            result = {"s": np.asarray(ref_ring.get_s_pos(refpts), dtype=float), "reference_kind": "run_input_lattice"}
            for plane, column in (("x", 0), ("y", 1)):
                result[f"beta_beating_{plane}"] = np.divide(
                    fit_beta[:, column] - ref_beta[:, column], ref_beta[:, column],
                    out=np.full(ref_beta.shape[0], np.nan), where=ref_beta[:, column] != 0,
                )
            return result
        except Exception:
            return None

    @property
    def svd_metadata(self) -> dict[str, Any]:
        config = self.options
        spectrum = self._npz_value("singular_values")
        return {
            "method": config.get("svd_selection_method"), "threshold": config.get("svd_threshold"),
            "rank": config.get("cut_"), "spectrum": spectrum,
            "measurement_values": int(self.fitted_orm.size) if self.fitted_orm is not None else None,
            "fitted_dofs": self.fitted_parameter_count,
        }

    @property
    def partitions(self) -> OrmPartitions | None:
        """Return pyLOCO's saved ORM partition: rows [H BPM, V BPM], columns [H cor, V cor]."""
        mapping = self.request.get("backend_mapping", {}).get("MachineElements", {})
        hbpm = len(mapping.get("bpm_ords") or [])
        hcor = len(mapping.get("horizontal_corrector_ords") or [])
        vcor = len(mapping.get("vertical_corrector_ords") or [])
        matrix = self.fitted_orm
        if matrix is None:
            matrix = self.initial_orm
        if matrix is not None:
            rows, cols = matrix.shape
            if hbpm == 0 or 2 * hbpm != rows:
                hbpm = rows // 2
            if hcor + vcor != cols:
                hcor = cols // 2
                vcor = cols - hcor
        if hbpm <= 0 or hcor + vcor <= 0:
            return None
        return OrmPartitions(hbpm, hbpm, hcor, vcor)

    @property
    def orm_raw_limits(self) -> tuple[float, float] | None:
        """Common full-resolution limits for measured/initial/fitted ORM views."""
        return self._combined_limits((self.measured_orm, self.initial_orm, self.fitted_orm), symmetric=False)

    @property
    def orm_residual_limit(self) -> float | None:
        """Common symmetric full-resolution limit for before/after residual views."""
        limits = self._combined_limits((self.residual_before, self.residual_after), symmetric=True)
        return limits[1] if limits else None

    @property
    def orm_residual_limits(self) -> tuple[float, float] | None:
        """Common full-resolution limits matching the standard viridis ORM plots."""
        return self._combined_limits((self.residual_before, self.residual_after), symmetric=False)

    @property
    def measured_orm(self):
        if "measured_orm" not in self._cache:
            self._cache["measured_orm"] = self._load_measured_orm()
        return self._cache["measured_orm"]

    @property
    def initial_orm(self):
        if "initial_orm" not in self._cache:
            self._cache["initial_orm"] = self._hdf5_matrix(
                self.result_dir / "model_orm_initial.h5", "response_matrix", "initial_orm"
            )
        return self._cache["initial_orm"]

    @property
    def fitted_orm(self):
        if "fitted_orm" not in self._cache:
            value = self._npz_value("orm_model")
            self._cache["fitted_orm"] = _as_matrix(value)
            if self._cache["fitted_orm"] is None and "fitted_orm" not in self._unavailable:
                self._unavailable["fitted_orm"] = "loco_results.npz does not contain a two-dimensional 'orm_model' array"
        return self._cache["fitted_orm"]

    @property
    def residual_before(self):
        return self._residual("residual_before", self.initial_orm)

    @property
    def residual_after(self):
        return self._residual("residual_after", self.fitted_orm)

    def _residual(self, key: str, model):
        if key not in self._cache:
            measured = self.measured_orm
            self._cache[key] = measured - model if measured is not None and model is not None and measured.shape == model.shape else None
            if self._cache[key] is None:
                model_key = "initial_orm" if key == "residual_before" else "fitted_orm"
                if measured is None:
                    self._unavailable[key] = self._unavailable.get("measured_orm", "measured ORM is unavailable")
                elif model is None:
                    self._unavailable[key] = self._unavailable.get(model_key, f"{model_key} is unavailable")
                else:
                    self._unavailable[key] = f"matrix shapes differ: measured {measured.shape}, model {model.shape}"
        return self._cache[key]

    @property
    def orm_rms_before(self) -> float | None:
        return _rms(self.residual_before)

    @property
    def orm_rms_after(self) -> float | None:
        return _rms(self.residual_after)

    @property
    def orm_max_before(self) -> float | None:
        return _max_abs(self.residual_before)

    @property
    def orm_max_after(self) -> float | None:
        return _max_abs(self.residual_after)

    @property
    def orm_improvement_percent(self) -> float | None:
        before, after = self.orm_rms_before, self.orm_rms_after
        if before is None or after is None or before == 0:
            return None
        return 100.0 * (before - after) / before

    @property
    def generated_files(self) -> list[Path]:
        if not self.result_dir.exists():
            return []
        return sorted((path for path in self.result_dir.rglob("*") if path.is_file()), key=lambda path: str(path.relative_to(self.result_dir)))

    @property
    def diagnostics(self) -> list[tuple[str, str]]:
        result = [("success", "LOCO execution completed")]
        reduction = self.chi2_reduction_percent
        if reduction is not None:
            result.append(("success" if reduction >= 0 else "warning", f"χ² {'decreased' if reduction >= 0 else 'increased'} by {abs(reduction):.1f}%"))
        improvement = self.orm_improvement_percent
        if improvement is not None:
            result.append(("success" if improvement >= 0 else "warning", f"ORM residual RMS {'decreased' if improvement >= 0 else 'increased'} by {abs(improvement):.1f}%"))
        if not self.dispersion_included:
            result.append(("info", "Dispersion was not included in this fit"))
        return result

    def _npz_value(self, name: str):
        key = f"npz:{name}"
        if key not in self._cache:
            path = self.result_dir / "loco_results.npz"
            if not path.exists():
                self._cache[key] = None
            else:
                try:
                    import numpy as np
                    with np.load(path, allow_pickle=True) as archive:
                        self._cache[key] = np.array(archive[name]) if name in archive else None
                except (OSError, ValueError, KeyError):
                    self._cache[key] = None
        return self._cache[key]

    def _combined_limits(self, matrices, *, symmetric: bool):
        key = "residual_limits" if symmetric else "raw_limits"
        if key not in self._cache:
            import numpy as np
            values = []
            for matrix in matrices:
                if matrix is not None:
                    finite = np.asarray(matrix, dtype=float)
                    finite = finite[np.isfinite(finite)]
                    if finite.size:
                        values.append(finite)
            if not values:
                result = None
            else:
                joined = np.concatenate(values)
                if symmetric:
                    limit = float(np.max(np.abs(joined)))
                    result = (-limit, limit) if limit > 0 else (-1.0, 1.0)
                else:
                    low, high = float(np.min(joined)), float(np.max(joined))
                    result = (low, high) if low < high else (low - 1.0, high + 1.0)
            self._cache[key] = result
        return self._cache[key]

    def _hdf5_matrix(self, path: Path, dataset: str, quantity: str | None = None):
        if not path.exists():
            if quantity:
                self._unavailable[quantity] = f"file does not exist: {path}"
            return None
        try:
            import h5py
            with h5py.File(path, "r") as handle:
                if dataset not in handle:
                    if quantity:
                        self._unavailable[quantity] = f"HDF5 dataset {dataset!r} is missing from {path}"
                    return None
                matrix = _as_matrix(handle[dataset][()])
                if matrix is None and quantity:
                    self._unavailable[quantity] = f"HDF5 dataset {dataset!r} in {path} is not two-dimensional"
                return matrix
        except (OSError, ValueError) as exc:
            if quantity:
                self._unavailable[quantity] = f"could not read {path}: {exc}"
            return None

    def _load_measured_orm(self):
        value = self.request.get("measurements", {}).get("orm")
        if not value:
            self._unavailable["measured_orm"] = "run_request.json does not reference an ORM measurement file"
            return None
        path = self._resolve_reference(value)
        if path is None:
            self._unavailable["measured_orm"] = f"referenced measurement file could not be resolved: {value}"
            return None
        suffix = path.suffix.lower()
        try:
            import numpy as np
            if suffix in {".h5", ".hdf5"}:
                matrix = self._hdf5_matrix(path, "response_matrix", "measured_orm")
            elif suffix == ".npy":
                matrix = _as_matrix(np.load(path, allow_pickle=False))
            elif suffix == ".npz":
                with np.load(path, allow_pickle=False) as archive:
                    key = "response_matrix" if "response_matrix" in archive else archive.files[0]
                    matrix = _as_matrix(archive[key])
            elif suffix == ".mat":
                from scipy.io import loadmat
                arrays = {key: item for key, item in loadmat(path).items() if not key.startswith("__")}
                matrix = _as_matrix(arrays.get("response_matrix", next(iter(arrays.values()), None)))
            else:
                self._unavailable["measured_orm"] = f"unsupported measurement file type: {suffix}"
                return None
        except (OSError, ValueError, KeyError, ImportError) as exc:
            self._unavailable["measured_orm"] = f"could not load referenced measurement file {path}: {exc}"
            return None
        target = self.fitted_orm
        if target is None:
            target = self.initial_orm
        if matrix is not None and target is not None and matrix.shape != target.shape:
            bad = self.request.get("backend_mapping", {}).get("BadBPMPositions") or []
            if bad and matrix.shape[0] - 2 * len(bad) == target.shape[0] and matrix.shape[1] == target.shape[1]:
                matrix = np.delete(matrix, list(bad) + [matrix.shape[0] // 2 + int(i) for i in bad], axis=0)
        return matrix

    def _resolve_reference(self, value: str | Path) -> Path | None:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path if path.exists() else None
        project_path = self.request.get("project_path")
        candidates = []
        if project_path:
            candidates.append(Path(project_path).expanduser().resolve().parent / path)
        candidates.extend((self.result_dir / path, Path.cwd() / path))
        return next((candidate.resolve() for candidate in candidates if candidate.exists()), None)

    def _load_vector_reference(self, value, *, dataset_names=(), fallback_index=0):
        if not value:
            return None
        path = self._resolve_reference(value)
        if path is None:
            return None
        try:
            import numpy as np
            if path.suffix.lower() in {".h5", ".hdf5"}:
                import h5py
                with h5py.File(path, "r") as handle:
                    for name in dataset_names:
                        if name in handle:
                            return np.ravel(handle[name][()]).astype(float)
                    arrays = [np.asarray(item[()]) for item in handle.values() if hasattr(item, "shape")]
                    return np.ravel(arrays[fallback_index]).astype(float) if len(arrays) > fallback_index else None
            if path.suffix.lower() == ".npy":
                return np.ravel(np.load(path, allow_pickle=False)).astype(float)
            if path.suffix.lower() == ".npz":
                with np.load(path, allow_pickle=False) as archive:
                    return np.ravel(archive[archive.files[0]]).astype(float)
            if path.suffix.lower() == ".mat":
                from scipy.io import loadmat
                arrays = [v for k, v in loadmat(path).items() if not k.startswith("__")]
                return np.ravel(arrays[0]).astype(float) if arrays else None
        except (OSError, ValueError, ImportError, KeyError):
            return None
        return None


def _as_matrix(value):
    if value is None:
        return None
    import numpy as np
    array = np.asarray(value, dtype=float)
    return array if array.ndim == 2 else None


def _rms(value) -> float | None:
    if value is None:
        return None
    import numpy as np
    finite = np.asarray(value, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite ** 2))) if finite.size else None


def _max_abs(value) -> float | None:
    if value is None:
        return None
    import numpy as np
    finite = np.asarray(value, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(np.abs(finite))) if finite.size else None


def _finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    import math
    return result if math.isfinite(result) else None


def _scalar_value(value, default=None):
    if value is None:
        return default
    try:
        return value.item()
    except (AttributeError, ValueError):
        return value
