"""Transactional application of a frozen full-FIT B2 manifest.

Simulation only. PETRA hardware application is deliberately unsupported.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path

from .quadrupole_transaction import (
    SimulationConnection,
    close,
    require_supported_simulation_profile,
)


SCHEMA = "pyloco.fullfit_b2_preview.v1"
JOURNAL_SCHEMA = "pyloco.fullfit_b2_transaction.v1"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FullFitB2Transaction:
    def __init__(self, connection=None, journal_directory=None):
        self.connection = connection or SimulationConnection(diagnostics_port=13132)
        self.directory = Path(
            journal_directory or
            "Examples/Demo/fullfit_transaction_journals"
        )
        self.record = None
        self.path = None

    def _persist(self):
        self.directory.mkdir(parents=True, exist_ok=True)

        if self.path is None:
            self.path = self.directory / f"{uuid.uuid4()}.json"

        tmp = self.path.with_suffix(".tmp")

        with tmp.open("w") as f:
            json.dump(self.record, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, self.path)

    def _snapshot_rows(self, control=None):
        try:
            snapshot = self.connection.snapshot(control=control)
        except TypeError:
            # Test doubles and older connection objects expose only snapshot().
            snapshot = self.connection.snapshot()
        identity = snapshot["identity"]

        require_supported_simulation_profile(identity)

        rows = {}
        for row in snapshot["quadrupoles"]:
            name = row["control"]

            if name in rows:
                raise RuntimeError(
                    f"Duplicate live B2 control: {name}"
                )

            rows[name] = row

        return snapshot, rows

    @staticmethod
    def _verify_calibration(live, frozen):
        keys = (
            "factor",
            "offset",
            "current",
            "physical",
        )

        for live_key, frozen_key in (
            ("factor", "factor"),
            ("offset", "offset"),
            ("current", "current_control"),
            ("physical", "current_physical_k"),
        ):
            a = float(live[live_key])
            b = float(frozen[frozen_key])

            if not close(a, b):
                raise RuntimeError(
                    f"Frozen/live {live_key} mismatch for "
                    f"{frozen['control']}"
                )

        factor = float(live["factor"])
        offset = float(live["offset"])
        current = float(live["current"])
        physical = float(live["physical"])

        if not all(
            math.isfinite(v)
            for v in (factor, offset, current, physical)
        ):
            raise RuntimeError(
                f"Non-finite calibration for {frozen['control']}"
            )

        if factor == 0:
            raise RuntimeError(
                f"Zero calibration factor for {frozen['control']}"
            )

        if not close(physical, factor * current + offset):
            raise RuntimeError(
                f"Physical/control transformation mismatch for "
                f"{frozen['control']}"
            )

    def load_preview_manifest(self, path):
        manifest_path = Path(path).resolve()
        manifest = json.loads(manifest_path.read_text())

        if manifest.get("schema") != SCHEMA:
            raise ValueError("Unsupported full-FIT preview schema")

        if manifest.get("mode") != "READ_ONLY_PREVIEW":
            raise ValueError("Manifest is not a read-only preview")

        count = manifest.get("count")
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Manifest count must be a positive integer")

        if not math.isclose(
            float(manifest.get("fraction")),
            0.10,
            rel_tol=0,
            abs_tol=1e-15,
        ):
            raise ValueError("Only the validated 10% correction is enabled")

        records = manifest.get("records")
        if not isinstance(records, list) or len(records) != count:
            actual = (
                len(records)
                if isinstance(records, list)
                else 0
            )
            raise ValueError(
                f"Manifest count={count} but contains "
                f"{actual} records"
            )

        controls = [r["control"] for r in records]
        if len(set(controls)) != count:
            raise ValueError(
                "Manifest contains duplicate controls"
            )

        # Revalidate all pinned source artifacts.
        for key in ("source_lattice", "fitted_lattice", "mapping_file"):
            entry = manifest[key]
            if _sha256(entry["path"]) != entry["sha256"]:
                raise RuntimeError(f"{key} SHA256 mismatch")

        snapshot, live_rows = self._snapshot_rows()

        frozen_identity = manifest["server_identity"]
        live_identity = snapshot["identity"]

        if manifest["source_lattice"]["sha256"] != live_identity.get("lattice_sha256"):
            raise RuntimeError(
                "FIT source lattice does not match the selected pySC profile lattice"
            )

        # Require the same exact simulation instance used for Preview.
        if live_identity != frozen_identity:
            raise RuntimeError(
                "pySC server identity changed since Preview. "
                "Generate a fresh manifest."
            )

        for frozen in records:
            control = frozen["control"]

            if control not in live_rows:
                raise RuntimeError(
                    f"Frozen control absent from live server: {control}"
                )

            live = live_rows[control]

            if live["component"] != "B2" or live["unit"] != "m^-2":
                raise RuntimeError(
                    f"Unsupported live control: {control}"
                )

            self._verify_calibration(live, frozen)

            proposed = float(frozen["proposed_control"])
            expected = float(frozen["expected_physical_k"])

            # A cumulative transaction carries the exact state from before
            # the first Apply.  Older one-pass manifests naturally restore
            # to the current (previewed) state.
            frozen.setdefault("restore_control", frozen["current_control"])
            frozen.setdefault("restore_physical_k", frozen["current_physical_k"])
            restore_control = float(frozen["restore_control"])
            restore_physical = float(frozen["restore_physical_k"])

            if not all(math.isfinite(value) for value in (
                proposed, expected, restore_control, restore_physical
            )):
                raise RuntimeError(
                    f"Non-finite proposal for {control}"
                )

            if not close(
                restore_physical,
                float(frozen["factor"]) * restore_control + float(frozen["offset"]),
            ):
                raise RuntimeError(
                    f"Original physical/control transformation mismatch for {control}"
                )

        application_number = int(manifest.get("application_number", 1))
        cumulative_fraction = float(
            manifest.get("cumulative_fraction", application_number * 0.10)
        )
        if application_number <= 0 or not math.isclose(
            cumulative_fraction,
            application_number * 0.10,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("Invalid cumulative correction sequence")
        history = manifest.get("history", [])
        if not isinstance(history, list) or len(history) != application_number - 1:
            raise ValueError("Cumulative correction history is incomplete")
        control_set = set(controls)
        for state in history:
            values = state.get("values", {})
            if set(values) != control_set:
                raise ValueError("Cumulative correction history controls do not match")
            for frozen in records:
                value = values[frozen["control"]]
                control_value = float(value["control"])
                physical_value = float(value["physical"])
                if not all(math.isfinite(v) for v in (control_value, physical_value)):
                    raise ValueError("Non-finite cumulative correction history")
                if not close(
                    physical_value,
                    float(frozen["factor"]) * control_value + float(frozen["offset"]),
                ):
                    raise ValueError("Invalid cumulative history calibration")
        if history:
            latest = history[-1]["values"]
            for frozen in records:
                prior = latest[frozen["control"]]
                if not close(float(prior["control"]), float(frozen["current_control"])):
                    raise ValueError("Cumulative preview does not start at prior control state")
                if not close(float(prior["physical"]), float(frozen["current_physical_k"])):
                    raise ValueError("Cumulative preview does not start at prior physical state")

        self.record = {
            "schema": JOURNAL_SCHEMA,
            "status": "preview_verified",
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "identity": frozen_identity,
            "fraction": 0.10,
            "application_number": application_number,
            "cumulative_fraction": cumulative_fraction,
            "restore_required": bool(manifest.get("restore_required", False)),
            "history": history,
            "items": records,
        }

        self.path = None

        return self.record

    def verify_applied(self):
        """Verify and return the currently applied cumulative state."""
        if self.record is None or self.record.get("status") != "applied":
            raise RuntimeError("No applied cumulative correction to verify")
        snapshot, rows = self._snapshot_rows()
        if snapshot["identity"] != self.record["identity"]:
            raise RuntimeError("pySC server identity changed")
        for item in self.record["items"]:
            row = rows.get(item["control"])
            if row is None:
                raise RuntimeError(f"Control disappeared: {item['control']}")
            if not close(float(row["current"]), float(item["proposed_control"])):
                raise RuntimeError(f"Applied control changed: {item['control']}")
            if not close(float(row["physical"]), float(item["expected_physical_k"])):
                raise RuntimeError(f"Applied physical K changed: {item['control']}")
        return self.record

    def _verify_fresh(self):
        if self.record is None:
            raise RuntimeError("No verified manifest loaded")

        snapshot, rows = self._snapshot_rows()

        if snapshot["identity"] != self.record["identity"]:
            raise RuntimeError("pySC server identity changed")

        for frozen in self.record["items"]:
            control = frozen["control"]

            if control not in rows:
                raise RuntimeError(
                    f"Control disappeared: {control}"
                )

            self._verify_calibration(rows[control], frozen)

        return snapshot, rows

    def apply(self, *, confirmed=False, progress=None):
        if not confirmed:
            raise PermissionError(
                "Explicit confirmation is required"
            )

        if (
            self.record is None
            or self.record["status"] != "preview_verified"
        ):
            raise RuntimeError(
                "Fresh verified full-FIT preview required"
            )

        # Whole-transaction gate BEFORE first SET.
        self._verify_fresh()

        self.record["status"] = "write_pending"
        # The durable pre-write journal deliberately marks every item as
        # restorable. Recovery can therefore restore the exact original state
        # even if the process stops between per-item progress checkpoints.
        self.record["restore_required"] = True

        for item in self.record["items"]:
            item["attempted"] = False
            item["set_completed"] = False
            item["verified"] = False

        self._persist()

        interface = self.connection.interface(
            self.record["identity"]
        )

        try:
            total = len(self.record["items"])
            for index, item in enumerate(self.record["items"], start=1):
                # Reverify ONLY this not-yet-written item immediately
                # before its SET. Previously applied items are expected
                # to differ from their frozen original values.
                snapshot, rows = self._snapshot_rows(item["control"])

                if snapshot["identity"] != self.record["identity"]:
                    raise RuntimeError("pySC server identity changed")

                if item["control"] not in rows:
                    raise RuntimeError(
                        f"Control disappeared: {item['control']}"
                    )

                live = rows[item["control"]]
                self._verify_calibration(live, item)

                item["attempted"] = True
                if progress:
                    progress(index, total, item["control"], "writing")

                interface.set(
                    item["control"],
                    float(item["proposed_control"]),
                )

                item["set_completed"] = True

                after_snapshot, after_rows = self._snapshot_rows(item["control"])
                after = after_rows[item["control"]]

                control = float(after["current"])
                physical = float(after["physical"])

                item["applied_readback"] = {
                    "control": control,
                    "physical": physical,
                }

                if not close(
                    control,
                    float(item["proposed_control"]),
                ):
                    raise RuntimeError(
                        f"Control verification failed: "
                        f"{item['control']}"
                    )

                if not close(
                    physical,
                    float(item["expected_physical_k"]),
                ):
                    raise RuntimeError(
                        f"Physical K verification failed: "
                        f"{item['control']}"
                    )

                item["verified"] = True
                if progress:
                    progress(index, total, item["control"], "verified")

            # Final whole-machine verification.
            _, rows = self._snapshot_rows()

            for item in self.record["items"]:
                row = rows[item["control"]]

                if not close(
                    float(row["current"]),
                    float(item["proposed_control"]),
                ):
                    raise RuntimeError(
                        f"Final control verification failed: "
                        f"{item['control']}"
                    )

                if not close(
                    float(row["physical"]),
                    float(item["expected_physical_k"]),
                ):
                    raise RuntimeError(
                        f"Final physical verification failed: "
                        f"{item['control']}"
                    )

            self.record["status"] = "applied"
            self.record["restore_required"] = True
            self.record.setdefault("history", []).append({
                "application_number": self.record["application_number"],
                "cumulative_fraction": self.record["cumulative_fraction"],
                "values": {
                    item["control"]: {
                        "control": float(item["proposed_control"]),
                        "physical": float(item["expected_physical_k"]),
                    }
                    for item in self.record["items"]
                },
            })
            self._persist()
            return self.record

        except Exception as exc:
            self.record["apply_error"] = str(exc)

            try:
                self.restore()
            except Exception as restore_exc:
                raise RuntimeError(
                    f"Apply failed: {exc}; "
                    f"RESTORATION FAILED: {restore_exc}"
                ) from exc

            raise RuntimeError(
                f"Apply failed: {exc}; "
                "all original values restored and verified"
            ) from exc

    def undo_last(self):
        """Remove only the most recently applied cumulative increment."""
        if self.record is None or self.record.get("status") not in (
            "applied", "preview_verified"
        ):
            raise RuntimeError("No cumulative correction increment to undo")

        history = self.record.get("history", [])
        if not history:
            raise RuntimeError("No cumulative correction increment to undo")

        snapshot, rows = self._snapshot_rows()
        if snapshot["identity"] != self.record["identity"]:
            raise RuntimeError("pySC server identity changed")

        current_state = history[-1]["values"]
        for control, expected in current_state.items():
            row = rows.get(control)
            if row is None:
                raise RuntimeError(f"Control disappeared: {control}")
            if not close(float(row["current"]), float(expected["control"])):
                raise RuntimeError(f"Current control changed before Undo: {control}")
            if not close(float(row["physical"]), float(expected["physical"])):
                raise RuntimeError(f"Current physical K changed before Undo: {control}")

        target_state = (
            history[-2]["values"] if len(history) > 1 else {
                item["control"]: {
                    "control": float(item["restore_control"]),
                    "physical": float(item["restore_physical_k"]),
                }
                for item in self.record["items"]
            }
        )
        self.record["status"] = "undoing"
        self._persist()
        interface = self.connection.interface(self.record["identity"])

        try:
            for item in reversed(self.record["items"]):
                interface.set(
                    item["control"],
                    float(target_state[item["control"]]["control"]),
                )
            _, after_rows = self._snapshot_rows()
            for control, expected in target_state.items():
                row = after_rows[control]
                if float(row["current"]) != float(expected["control"]):
                    raise RuntimeError(f"Undo control verification failed: {control}")
                if not close(float(row["physical"]), float(expected["physical"])):
                    raise RuntimeError(f"Undo physical K verification failed: {control}")
        except Exception as exc:
            rollback_errors = []
            for control, expected in current_state.items():
                try:
                    interface.set(control, float(expected["control"]))
                except Exception as rollback_exc:
                    rollback_errors.append(f"{control}: {rollback_exc}")
            self.record["status"] = "applied"
            self.record["undo_error"] = str(exc)
            self.record["undo_rollback_errors"] = rollback_errors
            self._persist()
            detail = f"; rollback failures: {'; '.join(rollback_errors)}" if rollback_errors else ""
            raise RuntimeError(f"Undo failed and the current cumulative state was retained{detail}") from exc

        history.pop()
        remaining = len(history)
        for item in self.record["items"]:
            target = target_state[item["control"]]
            item["proposed_control"] = float(target["control"])
            item["expected_physical_k"] = float(target["physical"])
            item["applied_readback"] = dict(target)
        self.record["application_number"] = remaining
        self.record["cumulative_fraction"] = remaining * 0.10
        self.record["restore_required"] = remaining > 0
        self.record["status"] = "applied" if remaining else "restored"
        self.record["last_action"] = "undo_last_increment"
        self._persist()
        return self.record

    def restore(self):
        if self.record is None or self.record["status"] not in (
            "applied",
            "preview_verified",
            "write_pending",
            "restoring",
            "restore_failed",
        ) or not self.record.get("restore_required", self.record.get("status") != "preview_verified"):
            raise RuntimeError(
                "No written full-FIT transaction to restore"
            )

        self.record["status"] = "restoring"

        errors = []

        try:
            self._persist()
        except Exception as exc:
            errors.append(f"Journal: {exc}")

        interface = self.connection.interface(
            self.record["identity"]
        )

        for item in reversed(self.record["items"]):
            try:
                did_restore = bool(
                    item.get("attempted") or self.record.get("restore_required")
                )
                if did_restore:
                    interface.set(
                        item["control"],
                        float(item["restore_control"]),
                    )

                _, rows = self._snapshot_rows(item["control"])
                row = rows[item["control"]]

                if float(row["current"]) != float(
                    item["restore_control"]
                ):
                    raise RuntimeError(
                        "Original control not exactly restored"
                    )

                if not close(
                    float(row["physical"]),
                    float(item["restore_physical_k"]),
                ):
                    raise RuntimeError(
                        "Original physical K not restored"
                    )

                item["restoration_status"] = (
                    "restored"
                    if did_restore
                    else "unchanged_verified"
                )

            except Exception as exc:
                item["restoration_status"] = "failed"
                item["restore_error"] = str(exc)
                errors.append(
                    f"{item['control']}: {exc}"
                )

        self.record["status"] = (
            "restore_failed" if errors else "restored"
        )
        if not errors:
            self.record["restore_required"] = False
        self.record["restore_errors"] = errors
        self._persist()

        if errors:
            raise RuntimeError("; ".join(errors))

        return self.record
