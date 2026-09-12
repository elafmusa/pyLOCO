from __future__ import annotations

import hashlib
import json

import pytest

from pyLOCO.correct.fullfit_transaction import FullFitB2Transaction, SCHEMA


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeConnection:
    def __init__(self, lattice_hash):
        self.identity = {
            "profile": "petra3_realistic",
            "machine": "PETRA III",
            "scenario": "realistic_errors",
            "lattice_sha256": lattice_hash,
        }
        self.current = 1.0
        self.factor = 2.0
        self.offset = 0.1
        self.snapshot_requests = []

    def snapshot(self, control=None):
        self.snapshot_requests.append(control)
        return {
            "identity": dict(self.identity),
            "quadrupoles": [{
                "control": "Q0K2_7_1/B2",
                "component": "B2",
                "unit": "m^-2",
                "factor": self.factor,
                "offset": self.offset,
                "current": self.current,
                "physical": self.factor * self.current + self.offset,
            }],
        }

    def interface(self, identity):
        assert identity == self.identity
        owner = self

        class Interface:
            def set(self, control, value):
                assert control == "Q0K2_7_1/B2"
                owner.current = float(value)

        return Interface()


def _manifest(tmp_path, connection, application, current, root, history=None):
    source = tmp_path / "source.mat"
    fitted = tmp_path / "fitted.mat"
    mapping = tmp_path / "mapping.json"
    for path in (source, fitted, mapping):
        if not path.exists():
            path.write_text(path.name)
    physical = connection.factor * current + connection.offset
    delta = 0.2
    payload = {
        "schema": SCHEMA,
        "mode": "READ_ONLY_PREVIEW",
        "count": 1,
        "fraction": 0.10,
        "application_number": application,
        "cumulative_fraction": application * 0.10,
        "restore_required": application > 1,
        "history": list(history or []),
        "server_identity": connection.identity,
        "source_lattice": {"path": str(source), "sha256": _sha(source)},
        "fitted_lattice": {"path": str(fitted), "sha256": _sha(fitted)},
        "mapping_file": {"path": str(mapping), "sha256": _sha(mapping)},
        "records": [{
            "control": "Q0K2_7_1/B2",
            "lattice_ordinal": 7,
            "lattice_name": "Q0K2_7_1",
            "family": "Q0K2",
            "factor": connection.factor,
            "offset": connection.offset,
            "current_control": current,
            "current_physical_k": physical,
            "physical_delta": delta,
            "control_delta": delta / connection.factor,
            "proposed_control": current + delta / connection.factor,
            "expected_physical_k": physical + delta,
            "restore_control": root,
            "restore_physical_k": connection.factor * root + connection.offset,
        }],
    }
    path = tmp_path / f"preview_{application:02d}.json"
    path.write_text(json.dumps(payload))
    return path


def test_two_ten_percent_passes_accumulate_and_restore_first_baseline(tmp_path):
    source = tmp_path / "source.mat"
    source.write_text(source.name)
    connection = FakeConnection(_sha(source))

    first = FullFitB2Transaction(connection, tmp_path / "journals")
    first.load_preview_manifest(_manifest(tmp_path, connection, 1, 1.0, 1.0))
    first.apply(confirmed=True)
    assert connection.current == pytest.approx(1.1)
    assert connection.snapshot()["quadrupoles"][0]["physical"] == pytest.approx(2.3)
    assert first.verify_applied()["cumulative_fraction"] == pytest.approx(0.1)

    second = FullFitB2Transaction(connection, tmp_path / "journals")
    second.load_preview_manifest(
        _manifest(tmp_path, connection, 2, 1.1, 1.0, first.record["history"])
    )
    second.apply(confirmed=True)
    assert connection.current == pytest.approx(1.2)
    assert connection.snapshot()["quadrupoles"][0]["physical"] == pytest.approx(2.5)
    assert second.record["cumulative_fraction"] == pytest.approx(0.2)

    second.undo_last()
    assert connection.current == pytest.approx(1.1)
    assert connection.snapshot()["quadrupoles"][0]["physical"] == pytest.approx(2.3)
    assert second.record["cumulative_fraction"] == pytest.approx(0.1)
    assert second.record["status"] == "applied"

    second.undo_last()
    assert connection.current == 1.0
    assert connection.snapshot()["quadrupoles"][0]["physical"] == pytest.approx(2.1)
    assert second.record["status"] == "restored"

    # Rebuild the two-step state to retain coverage of full restoration.
    first = FullFitB2Transaction(connection, tmp_path / "journals")
    first.load_preview_manifest(_manifest(tmp_path, connection, 1, 1.0, 1.0))
    first.apply(confirmed=True)
    second = FullFitB2Transaction(connection, tmp_path / "journals")
    second.load_preview_manifest(
        _manifest(tmp_path, connection, 2, 1.1, 1.0, first.record["history"])
    )
    second.apply(confirmed=True)

    second.restore()
    assert connection.current == 1.0
    assert connection.snapshot()["quadrupoles"][0]["physical"] == pytest.approx(2.1)
    assert second.record["status"] == "restored"
    assert second.record["restore_required"] is False


def test_unapplied_next_preview_can_restore_prior_cumulative_state(tmp_path):
    source = tmp_path / "source.mat"
    source.write_text(source.name)
    connection = FakeConnection(_sha(source))
    connection.current = 1.1
    transaction = FullFitB2Transaction(connection, tmp_path / "journals")
    history = [{
        "application_number": 1,
        "cumulative_fraction": 0.1,
        "values": {"Q0K2_7_1/B2": {"control": 1.1, "physical": 2.3}},
    }]
    transaction.load_preview_manifest(
        _manifest(tmp_path, connection, 2, 1.1, 1.0, history)
    )
    transaction.restore()
    assert connection.current == 1.0
    assert transaction.record["status"] == "restored"


def test_apply_uses_targeted_readback_and_only_durable_transaction_boundaries(tmp_path):
    source = tmp_path / "source.mat"
    source.write_text(source.name)
    connection = FakeConnection(_sha(source))
    transaction = FullFitB2Transaction(connection, tmp_path / "journals")
    transaction.load_preview_manifest(_manifest(tmp_path, connection, 1, 1.0, 1.0))

    connection.snapshot_requests.clear()
    persist_count = 0
    original_persist = transaction._persist

    def counted_persist():
        nonlocal persist_count
        persist_count += 1
        original_persist()

    transaction._persist = counted_persist
    progress = []
    transaction.apply(
        confirmed=True,
        progress=lambda *event: progress.append(event),
    )

    control = "Q0K2_7_1/B2"
    assert connection.snapshot_requests == [None, control, control, None]
    assert persist_count == 2
    assert progress == [
        (1, 1, control, "writing"),
        (1, 1, control, "verified"),
    ]
