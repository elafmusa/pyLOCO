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


def _anchored_manifest(tmp_path, connection, application, previous_fraction, target_fraction,
                       baseline=1.0, full_delta=.2, history=None):
    connection.current=baseline+(full_delta*previous_fraction)/connection.factor
    path=_manifest(tmp_path,connection,application,connection.current,baseline,history)
    data=json.loads(path.read_text())
    increment=target_fraction-previous_fraction
    data["fraction"]=increment
    data["previous_fraction"]=previous_fraction
    data["cumulative_fraction"]=target_fraction
    row=data["records"][0]
    row["full_physical_delta"]=full_delta
    row["physical_delta"]=full_delta*increment
    row["control_delta"]=full_delta*increment/connection.factor
    row["proposed_control"]=baseline+(full_delta*target_fraction)/connection.factor
    row["expected_physical_k"]=connection.factor*baseline+connection.offset+full_delta*target_fraction
    path.write_text(json.dumps(data))
    return path


def test_zero_to_thirty_percent_targets_are_anchored_to_original_baseline(tmp_path):
    source=tmp_path/"source.mat"; source.write_text(source.name)
    connection=FakeConnection(_sha(source)); history=[]
    for application,(start,target) in enumerate(((0,.1),(.1,.2),(.2,.3)),start=1):
        transaction=FullFitB2Transaction(connection,tmp_path/"journals")
        transaction.load_preview_manifest(_anchored_manifest(
            tmp_path,connection,application,start,target,history=history))
        transaction.apply(confirmed=True)
        history=list(transaction.record["history"])
        assert connection.current==pytest.approx(1.0+(.2*target)/2.0)
        assert transaction.record["cumulative_fraction"]==pytest.approx(target)
        assert history[-1]["execution_status"]=="verified"
        assert history[-1]["requested_target_fraction"]==pytest.approx(target)
        with pytest.raises(RuntimeError,match="Fresh verified"):
            transaction.apply(confirmed=True)
    assert [step["cumulative_fraction"] for step in history]==pytest.approx([.1,.2,.3])


def test_restart_reconciles_readbacks_and_rejects_stale_machine_state(tmp_path):
    source=tmp_path/"source.mat"; source.write_text(source.name)
    connection=FakeConnection(_sha(source))
    transaction=FullFitB2Transaction(connection,tmp_path/"journals")
    transaction.load_preview_manifest(_anchored_manifest(tmp_path,connection,1,0,.1))
    transaction.apply(confirmed=True)
    resumed=FullFitB2Transaction(connection,tmp_path/"journals")
    assert resumed.load_journal(transaction.path)["cumulative_fraction"]==pytest.approx(.1)
    connection.current+=.01
    stale=FullFitB2Transaction(connection,tmp_path/"journals")
    with pytest.raises(RuntimeError,match="Applied control changed"):
        stale.load_journal(transaction.path)


def test_restore_uses_original_baseline_after_three_steps(tmp_path):
    source=tmp_path/"source.mat"; source.write_text(source.name)
    connection=FakeConnection(_sha(source)); history=[]; transaction=None
    for application,(start,target) in enumerate(((0,.1),(.1,.2),(.2,.3)),start=1):
        transaction=FullFitB2Transaction(connection,tmp_path/"journals")
        transaction.load_preview_manifest(_anchored_manifest(tmp_path,connection,application,start,target,history=history))
        transaction.apply(confirmed=True); history=list(transaction.record["history"])
    transaction.restore()
    assert connection.current==pytest.approx(1.0)
    assert transaction.record["status"]=="restored"


def test_interrupted_journal_requires_explicit_recovery(tmp_path):
    source=tmp_path/"source.mat"; source.write_text(source.name)
    connection=FakeConnection(_sha(source))
    transaction=FullFitB2Transaction(connection,tmp_path/"journals")
    transaction.load_preview_manifest(_anchored_manifest(tmp_path,connection,1,0,.1))
    transaction.record["status"]="write_pending"; transaction.record["restore_required"]=True
    transaction._persist()
    resumed=FullFitB2Transaction(connection,tmp_path/"journals")
    with pytest.raises(RuntimeError,match="explicit recovery"):
        resumed.load_journal(transaction.path)


def test_failed_command_does_not_advance_verified_history_and_restores_baseline(tmp_path):
    source=tmp_path/"source.mat"; source.write_text(source.name)
    connection=FakeConnection(_sha(source))
    transaction=FullFitB2Transaction(connection,tmp_path/"journals")
    transaction.load_preview_manifest(_anchored_manifest(tmp_path,connection,1,0,.1))
    calls=0
    original_interface=connection.interface
    def faulty_interface(identity):
        interface=original_interface(identity)
        original_set=interface.set
        def set_value(control,value):
            nonlocal calls
            calls+=1
            original_set(control, value+.01 if calls==1 else value)
        interface.set=set_value
        return interface
    connection.interface=faulty_interface
    with pytest.raises(RuntimeError,match="all original values restored"):
        transaction.apply(confirmed=True)
    assert connection.current==pytest.approx(1.0)
    assert transaction.record["status"]=="restored"
    assert transaction.record["history"]==[]


def test_verified_step_can_be_marked_measured_and_persists(tmp_path):
    source=tmp_path/"source.mat"; source.write_text(source.name)
    measurement=tmp_path/"orm.h5"; measurement.write_bytes(b"measurement")
    connection=FakeConnection(_sha(source))
    transaction=FullFitB2Transaction(connection,tmp_path/"journals")
    transaction.load_preview_manifest(_anchored_manifest(tmp_path,connection,1,0,.1))
    transaction.apply(confirmed=True)
    transaction.mark_latest_measured([measurement])
    persisted=json.loads(transaction.path.read_text())
    assert persisted["history"][-1]["execution_status"]=="measured"
    assert persisted["history"][-1]["measurements"]["files"]==[str(measurement.resolve())]
