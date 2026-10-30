"""Durable, individual-magnet integrated-strength correction transactions."""
from __future__ import annotations

import copy
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .pyaml_connection import file_hash, resolve_mapping, review_identity


STRENGTH_GET_ABS_TOLERANCE = 1e-7  # integrated strength [m^-1]
STRENGTH_GET_REL_TOLERANCE = 1e-8
STRENGTH_GET_ATTEMPTS = 6
STRENGTH_GET_INTERVAL_S = 0.1


def close(actual, expected):
    """Compare pyAML adapter values at realistic control-system precision."""
    return math.isclose(actual, expected, rel_tol=STRENGTH_GET_REL_TOLERANCE,
                        abs_tol=STRENGTH_GET_ABS_TOLERANCE)


class PyAMLCorrectionTransaction:
    def __init__(self, connection, directory):
        self.connection = connection
        self.directory = Path(directory)
        self.record = None
        self.path = None
        self.mapping_path = None
        self._record_before_preview = None
        self._path_before_preview = None

    @property
    def pending(self):
        return bool(self.record and self.record["restore_required"])

    def persist(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.path is None:
            self.path = self.directory / f"{uuid4()}.json"
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(self.record, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
        if os.name == "posix":
            descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def identity(self, review, explicit):
        rows = resolve_mapping(review, self.connection.discover(), explicit)
        mapping = None if self.mapping_path is None else dict(path=str(Path(self.mapping_path).resolve()), sha256=file_hash(self.mapping_path))
        return copy.deepcopy(dict(connection=self.connection.identity(), review=review_identity(review, rows), explicit=explicit, mapping_file=mapping))

    def _verified_get(self, name, expected):
        """Poll the public strength.get() adapter value after a write.

        pyAML backends may publish a set asynchronously or quantize the stored
        integrated strength.  This remains a get()-based verification; it does
        not claim that an independent hardware readback exists.
        """
        actual = self.connection.read(name)
        for _ in range(STRENGTH_GET_ATTEMPTS - 1):
            if close(actual, expected):
                return actual
            time.sleep(STRENGTH_GET_INTERVAL_S)
            actual = self.connection.read(name)
        if close(actual, expected):
            return actual
        tolerance=max(STRENGTH_GET_ABS_TOLERANCE,
                      abs(expected)*STRENGTH_GET_REL_TOLERANCE)
        raise RuntimeError(
            f"Adapter strength.get() mismatch: {name}; requested {expected:.12g}, "
            f"get() returned {actual:.12g}, difference {actual-expected:+.6g}, "
            f"allowed tolerance {tolerance:.3g} m^-1"
        )

    def preview(self, review, explicit=()):
        explicit = list(explicit)
        identity = self.identity(review, explicit)
        step_fraction = float(review.global_scale)
        if not math.isfinite(step_fraction) or not 0 < step_fraction <= 1:
            raise ValueError("pyAML correction fraction must be greater than 0% and at most 100%")
        previous = self.record if self.pending else None
        if previous:
            if previous["status"] != "applied":
                raise RuntimeError("Apply the existing preview or restore before another preview")
            if previous["identity"] != identity:
                raise RuntimeError("Correction changed; restore original strengths first")
        fraction = (previous["fraction"] if previous else 0) + step_fraction
        if fraction > 1 and not math.isclose(fraction, 1, rel_tol=0, abs_tol=1e-14):
            raise ValueError("Next pass exceeds 100% cumulative correction; restore before starting another sequence")
        fraction = min(fraction, 1.0)
        rows = []
        for row in identity["review"]["rows"]:
            if not row["included"]:
                continue
            if row["status"] != "mapped":
                raise ValueError(f"Unresolved included correction: {row['lattice_name']} ({row['status']})")
            if type(row["lattice_ordinal"]) is not int or row["lattice_ordinal"] < 0:
                raise ValueError(f"Missing source lattice ordinal: {row['lattice_name']}")
            length = row["length"]
            if length is None or not math.isfinite(length) or length <= 0:
                raise ValueError(f"Invalid magnetic length: {row['lattice_name']}")
            name = row["magnet_name"]
            current = self.connection.read(name)
            prior = next((item for item in previous["changes"] if item["name"] == name), None) if previous else None
            if prior and not close(current, prior["target"]):
                raise RuntimeError(f"Stale adapter strength.get() value: {name}")
            original = prior["original"] if prior else current
            applied_delta_k = (row["delta"] / step_fraction) * fraction
            applied_delta_kl = applied_delta_k * length
            full_delta_kl = (row["delta"] / step_fraction) * length
            target = original + applied_delta_kl
            if not math.isfinite(target):
                raise ValueError(f"Nonfinite target: {name}")
            rows.append(dict(name=name, original=original, current=current, target=target,
                             readback=current, verification_evidence="adapter_value",
                             verification_source="magnet.strength.get()",
                             status="preview", component="B2L" if row["correction_type"] == "normal_quadrupole" else "A2L",
                             unit="m^-1", magnetic_length_m=length,
                             full_delta_kl_m1=full_delta_kl,
                             applied_delta_k_m2=applied_delta_k,
                             applied_delta_kl_m1=applied_delta_kl))
        if not rows:
            raise ValueError("No included corrections")
        self._record_before_preview = copy.deepcopy(previous)
        self._path_before_preview = self.path
        if not previous:
            self.path = None
        history=copy.deepcopy(previous.get("history",[])) if previous else []
        self.record = dict(schema="pyloco.pyaml_correction.v1", identity=identity, fraction=fraction,
                           previous_fraction=(previous["fraction"] if previous else 0.0),
                           increment_fraction=step_fraction, history=history,
                           status="preview", restore_required=bool(previous), changes=rows,
                           attempted=list(previous["attempted"]) if previous else [])
        self.persist()
        return self.record

    def cancel_preview(self):
        if not self.record or self.record.get("status") != "preview":
            raise RuntimeError("There is no read-only preview to cancel")
        cancelled = self.record
        prior = copy.deepcopy(self._record_before_preview)
        prior_path = self._path_before_preview
        if prior is None:
            # Retain an audit record that this preview made no writes, while
            # clearing it from the active transaction state.
            cancelled["status"] = "cancelled"
            cancelled["restore_required"] = False
            self.persist()
            self.record = None
            self.path = None
        else:
            self.record = prior
            self.path = prior_path
            self.persist()
        self._record_before_preview = None
        self._path_before_preview = None
        return self.record

    def apply(self, review, explicit=(), *, confirmed=False, cancel=None):
        if not confirmed:
            raise PermissionError("Explicit Apply confirmation required")
        if not self.connection.allow_writes:
            raise PermissionError("pyAML writes disabled")
        if not self.record or self.record["status"] != "preview":
            raise RuntimeError("Fresh preview required")
        if self.identity(review, list(explicit)) != self.record["identity"]:
            raise RuntimeError("Stale preview: configuration, mapping or correction changed")
        for row in self.record["changes"]:
            if not close(self.connection.read(row["name"]), row["current"]):
                raise RuntimeError(f"Stale preview adapter strength.get() value: {row['name']}")
        try:
            for row in self.record["changes"]:
                if cancel and cancel.is_set():
                    raise RuntimeError("Correction cancelled")
                self.connection.identity()
                self.record["status"] = "write_pending"
                self.record["restore_required"] = True
                if row["name"] not in self.record["attempted"]:
                    self.record["attempted"].append(row["name"])
                row["status"] = "write_pending"
                self.persist()
                self.connection.write(row["name"], row["target"])
                row["readback"] = self._verified_get(row["name"], row["target"])
                row["status"] = "applied"
                self.persist()
            if cancel and cancel.is_set():
                raise RuntimeError("Correction cancelled")
            self.record["status"] = "applied"
            self.record.setdefault("history",[]).append(dict(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                cumulative_fraction=self.record["fraction"],
                increment_fraction=self.record.get("increment_fraction",0.0),
                execution_status="verified", measurements={},
            ))
            self.persist()
        except Exception as exc:
            self.record["error"] = str(exc)
            if not self.pending:
                self.record["status"] = "cancelled"
                self.persist()
                raise
            try:
                self.restore()
            except Exception as restore_error:
                raise RuntimeError(f"{exc}; restoration failed: {restore_error}") from exc
            raise RuntimeError(f"{exc}; original strengths restored") from exc
        return self.record

    def mark_latest_measured(self, paths):
        if not self.record or self.record.get("status") != "applied" or not self.record.get("history"):
            raise RuntimeError("No verified pyAML correction step is available")
        self.record["history"][-1]["measurements"]={"files":[str(Path(path).resolve()) for path in paths]}
        self.record["history"][-1]["execution_status"]="measured"
        self.persist()
        return self.record

    def _verified_step_stack(self):
        """Return the still-applied increments represented by the audit history."""
        history = self.record.get("history", ()) if self.record else ()
        steps = []
        for event in history:
            status = event.get("execution_status")
            increment = float(event.get("increment_fraction", 0.0))
            if status == "undo_verified":
                if steps:
                    steps.pop()
            elif increment > 0 and status in {"verified", "measured"}:
                steps.append(increment)
        if not history and self.record and float(self.record.get("fraction", 0.0)) > 0:
            # Compatibility with journals written before per-step history was
            # introduced: the current cumulative fraction is one undoable step.
            steps.append(float(self.record["fraction"]))
        return steps

    def undo_last(self):
        """Undo one verified correction increment, never the whole baseline implicitly.

        Every live value must still match the latest verified target.  If a
        write or verification fails, magnets already touched by the undo are
        returned to their pre-undo targets and verified before the error is
        reported.
        """
        if not self.record or self.record.get("status") != "applied":
            raise RuntimeError("No verified correction step is available to undo")
        if not self.connection.allow_writes:
            raise PermissionError("pyAML writes disabled")
        if self.connection.identity() != self.record["identity"]["connection"]:
            raise RuntimeError("Connection identity mismatch; undo blocked")
        steps = self._verified_step_stack()
        if not steps:
            raise RuntimeError("No verified correction step is available to undo")
        increment = steps[-1]
        current_fraction = float(self.record["fraction"])
        target_fraction = max(0.0, current_fraction - increment)
        before = copy.deepcopy(self.record)
        targets = {}
        for row in self.record["changes"]:
            live = self.connection.read(row["name"])
            if not close(live, row["target"]):
                raise RuntimeError(
                    f"Stale adapter strength.get() value before undo: {row['name']}; "
                    f"expected {row['target']:.12g}, returned {live:.12g}"
                )
            targets[row["name"]] = (
                float(row["original"])
                + float(row["full_delta_kl_m1"]) * target_fraction
            )
        self.record["status"] = "undo_pending"
        self.persist()
        touched = []
        try:
            for row in self.record["changes"]:
                name = row["name"]
                touched.append(name)
                self.connection.write(name, targets[name])
                row["readback"] = self._verified_get(name, targets[name])
                row["status"] = "undo_verified"
                self.persist()
        except Exception as exc:
            rollback_errors = []
            prior = {row["name"]: row["target"] for row in before["changes"]}
            for name in reversed(touched):
                try:
                    self.connection.write(name, prior[name])
                    self._verified_get(name, prior[name])
                except Exception as rollback_exc:
                    rollback_errors.append(f"{name}: {rollback_exc}")
            if rollback_errors:
                self.record["status"] = "restore_failed"
                self.record["restore_required"] = True
                self.record["restoration_errors"] = rollback_errors
                self.persist()
                raise RuntimeError(
                    f"Undo failed: {exc}; pre-undo rollback failed: "
                    + "; ".join(rollback_errors)
                ) from exc
            self.record = before
            self.record["error"] = f"Undo failed and pre-undo state was restored: {exc}"
            self.persist()
            raise RuntimeError(
                f"Undo failed: {exc}; pre-undo correction state restored"
            ) from exc

        for row in self.record["changes"]:
            row["current"] = targets[row["name"]]
            row["target"] = targets[row["name"]]
            row["status"] = "applied" if target_fraction > 0 else "restored"
        self.record["fraction"] = target_fraction
        self.record["previous_fraction"] = target_fraction
        self.record["increment_fraction"] = 0.0
        self.record["status"] = "applied" if target_fraction > 0 else "restored"
        self.record["restore_required"] = target_fraction > 0
        self.record.setdefault("history", []).append(dict(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            cumulative_fraction=target_fraction,
            increment_fraction=-increment,
            execution_status="undo_verified",
            measurements={},
        ))
        self.record.pop("error", None)
        self.persist()
        return self.record

    def restore(self):
        if not self.pending:
            raise RuntimeError("No pending restoration")
        try:
            if self.connection.identity() != self.record["identity"]["connection"]:
                raise RuntimeError("Connection identity mismatch; restoration blocked")
        except Exception as exc:
            self.record["status"] = "restore_failed"
            self.record["restoration_errors"] = [str(exc)]
            self.persist()
            raise
        self.record["status"] = "restoring"
        self.persist()
        failures = []
        for row in reversed(self.record["changes"]):
            if row["name"] not in self.record["attempted"]:
                continue
            try:
                self.connection.write(row["name"], row["original"])
                row["readback"] = self._verified_get(row["name"], row["original"])
                row["status"] = "restored"
            except Exception as exc:
                row["status"] = "restore_failed"
                failures.append(f"{row['name']}: {exc}")
            self.persist()
        self.record["status"] = "restore_failed" if failures else "restored"
        self.record["restore_required"] = bool(failures)
        self.record["restoration_errors"] = failures
        self.persist()
        if failures:
            raise RuntimeError("; ".join(failures))
        return self.record

    def recover(self, path):
        if self.pending:
            raise RuntimeError("Restore the current transaction first")
        if not self.connection.allow_writes:
            raise PermissionError("Reconnect with magnet writes enabled before recovery")
        record = json.loads(Path(path).read_text())
        if record.get("schema") != "pyloco.pyaml_correction.v1" or not record.get("restore_required"):
            raise ValueError("Not a pending pyAML transaction journal")
        if record["identity"]["connection"] != self.connection.identity():
            raise ValueError("Journal belongs to a different pyAML connection")
        names = set()
        for row in record["changes"]:
            if row["name"] in names:
                raise ValueError("Duplicate journal target")
            names.add(row["name"])
            self.connection.read(row["name"])
            if not all(math.isfinite(float(row[key])) for key in ("original", "current", "target")):
                raise ValueError("Nonfinite journal strengths")
        if not set(record["attempted"]) <= names:
            raise ValueError("Invalid attempted targets")
        self.record = copy.deepcopy(record)
        self.record["status"] = "recovery_required"
        self.path = Path(path)
        return self.record
