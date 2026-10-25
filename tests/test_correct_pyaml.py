import json
from threading import Event
from types import SimpleNamespace

import pytest

from pyLOCO.correct.model import CorrectItem, CorrectionReview
from pyLOCO.correct.pyaml_connection import (PyAMLCorrectionConnection, resolve_mapping,
    load_mapping, mapping_template, validate_mapping_entries,
    validate_mapping_document)
from pyLOCO.correct.pyaml_transaction import PyAMLCorrectionTransaction


class Strength:
    def __init__(self, value):
        self.value = value
        self.writes = []
        self.fail = None
        self.after_write = None

    def get(self):
        return self.value

    def set(self, value):
        self.writes.append(value)
        self.value = value
        if self.after_write:
            self.after_write()
        if self.fail and self.fail(value):
            raise RuntimeError("write failed after changing magnet")

    def unit(self):
        raise AssertionError("No unit metadata access allowed")


class Magnet:
    def __init__(self, identity, value):
        self.identity = identity
        self.strength = Strength(value)

    def get_lattice_names(self):
        return self.identity


class Inventory:
    def __init__(self, names):
        self.members = names

    def names(self):
        return list(self.members)

    def __getitem__(self, index):
        raise AssertionError("No magnet array indexing")

    @property
    def strengths(self):
        raise AssertionError("No magnet array reads or writes")


@pytest.fixture
def setup(tmp_path):
    config = tmp_path / "machine.yaml"; config.write_text("test configuration")
    source = tmp_path / "source.json"; source.write_text("{}")
    magnets = {"M1": Magnet("Q1", 1), "M2": Magnet("Q2", 2), "S1": Magnet("SQ1", 3)}
    machine = SimpleNamespace(magnet=magnets, magnets={"Quads": Inventory(["M2", "M1"]), "SkewQuads": Inventory(["S1"])})
    connection = PyAMLCorrectionConnection(config, accelerator=SimpleNamespace(live=machine))
    items = [CorrectItem(index=index, correction_type=family, name=name, lattice_ordinal=index,
        unit="ignored", initial_value=1, fitted_value=.8, raw_fitted_delta=-.2,
        recommended_machine_delta=delta, sign_convention="initial - fitted",
        metadata={"magnetic_length_m": length})
        for index, (family, name, delta, length) in enumerate([
            ("normal_quadrupole", "Q1", .2, .5), ("normal_quadrupole", "Q2", -.1, .4),
            ("skew_quadrupole", "SQ1", .3, .2)])]
    review = CorrectionReview(items, str(source))
    transaction = PyAMLCorrectionTransaction(connection, tmp_path / "journals")
    return SimpleNamespace(connection=connection, review=review, transaction=transaction,
                           magnets=magnets, machine=machine, config=config, source=source)


def test_discovery_and_identity_matching_not_position(setup):
    rows = resolve_mapping(setup.review, setup.connection.inventory)
    assert [row["magnet_name"] for row in rows] == ["M1", "M2", "S1"]
    assert all(not magnet.strength.writes for magnet in setup.magnets.values())
    setup.review.items.reverse()
    setup.review.items[0].included = False
    rows = resolve_mapping(setup.review, setup.connection.inventory)
    assert [row["magnet_name"] for row in rows] == ["S1", "M2", "M1"]


def test_preview_application_scope_can_select_only_normal_or_skew(setup):
    setup.review.application_scope="normal_quadrupole"
    record=setup.transaction.preview(setup.review)
    assert {row["name"] for row in record["changes"]}=={"M1","M2"}
    setup.transaction.record=None
    setup.review.application_scope="skew_quadrupole"
    record=setup.transaction.preview(setup.review)
    assert {row["name"] for row in record["changes"]}=={"S1"}
    assert all(item.included for item in setup.review.items)


def test_identity_matching_accepts_exact_at_device_and_backend_aliases(setup):
    item = setup.review.items[0]
    item.name = "QF6D"
    item.metadata.update(lattice_fam_name="QF6D", lattice_device="srmag/m-qf6/c01-d")
    setup.magnets["M1"].identity = "srmag/m-qf6/c01-d"
    rows = resolve_mapping(setup.review, setup.connection.discover())
    assert rows[0]["status"] == "mapped"
    assert rows[0]["magnet_name"] == "M1"

    setup.magnets["M1"].identity = ["unrelated-alias", "srmag/m-qf6/c01-d"]
    rows = resolve_mapping(setup.review, setup.connection.discover())
    assert rows[0]["status"] == "mapped"
    assert rows[0]["magnet_name"] == "M1"


def test_identity_matching_uses_exact_control_name_without_terminal_component(setup):
    item=setup.review.items[0]
    item.name="QF6D"
    item.metadata["lattice_device"]="srmag/m-qf6/c01-d"
    magnet=setup.magnets.pop("M1")
    magnet.identity="unrelated-alias"
    control="srmag/m-qf6/c01-d-B2L"
    setup.magnets[control]=magnet
    setup.machine.magnets["Quads"].members=[
        control if name=="M1" else name for name in setup.machine.magnets["Quads"].members]
    rows=resolve_mapping(setup.review,setup.connection.discover())
    assert rows[0]["status"]=="mapped"
    assert rows[0]["magnet_name"]==control


@pytest.mark.parametrize("names", [["M1", "M1"], ["M1", "S1"]])
def test_duplicate_arrays_rejected(setup, names):
    setup.machine.magnets["Quads"].members = names
    with pytest.raises(ValueError, match="Duplicate"):
        setup.connection.discover()


def test_missing_and_empty_arrays(setup):
    setup.machine.magnets["SkewQuads"].members = []
    assert len(setup.connection.discover()) == 2
    setup.machine.magnets.pop("Quads")
    with pytest.raises(ValueError, match="Missing pyAML array"):
        setup.connection.discover()


def test_selectors_unmatched_and_explicit_mapping(setup, tmp_path):
    setup.magnets["M1"].identity = "Q*@0"
    inventory = setup.connection.discover()
    assert resolve_mapping(setup.review, inventory)[0]["status"] == "unmapped"
    template = mapping_template(setup.review)
    assert [row["control_name"] for row in template["mappings"]] == ["", "", ""]
    assert all(set(row) == {"lattice_name", "lattice_ordinal", "control_name"}
               for row in template["mappings"])
    for row, name in zip(template["mappings"], ["M1", "M2", "S1"]):
        row["control_name"] = name
    path = tmp_path / "mapping.yaml"
    import yaml
    path.write_text(yaml.safe_dump(template))
    explicit = load_mapping(path)
    assert resolve_mapping(setup.review, inventory, explicit)[0]["match_source"] == "explicit"
    explicit[0]["control_name"] = "S1"
    with pytest.raises(ValueError, match="wrong-family"):
        resolve_mapping(setup.review, inventory, explicit)


def test_duplicate_identities_need_explicit_mapping(setup):
    setup.magnets["M2"].identity = "Q1"
    inventory = setup.connection.discover()
    assert resolve_mapping(setup.review, inventory)[0]["status"] == "ambiguous"
    entries = mapping_template(setup.review)["mappings"]
    for row, name in zip(entries, ["M1", "M2", "S1"]):
        row["control_name"] = name
    assert all(row["status"] == "mapped" for row in resolve_mapping(setup.review, inventory, entries))
    entries[1]["control_name"] = "M1"
    with pytest.raises(ValueError, match="Duplicate mapping target"):
        resolve_mapping(setup.review, inventory, entries)


def test_template_with_unmapped_fit_names_is_safe_and_not_a_mapping(setup):
    setup.review.items[0].name = "QD3J"
    template = mapping_template(setup.review)
    row = template["mappings"][0]
    assert row == {
        "lattice_ordinal": 0,
        "lattice_name": "QD3J",
        "control_name": "",
    }
    with pytest.raises(ValueError, match="wrong-family mapping target"):
        resolve_mapping(setup.review, setup.connection.inventory, [
            {**row, "control_name": "QD3J"}
        ])


def test_user_mapping_status_cannot_authorize_an_invalid_target(setup):
    row = mapping_template(setup.review)["mappings"][0]
    row.update(control_name="not-a-device", mapping_status="mapped")
    with pytest.raises(ValueError, match="wrong-family mapping target"):
        resolve_mapping(setup.review, setup.connection.inventory, [row])


def test_legacy_full_mapping_metadata_remains_supported(setup):
    rows = mapping_template(setup.review)["mappings"]
    names = ["M1", "M2", "S1"]
    for row, item, name in zip(rows, setup.review.items, names):
        row.update(
            control_name=name,
            component="B2L" if item.correction_type == "normal_quadrupole" else "A2L",
            unit="m^-1",
            mapping_status="mapped",
        )
    resolved = resolve_mapping(setup.review, setup.connection.inventory, rows)
    assert all(row["status"] == "mapped" for row in resolved)


def test_minimal_mapping_derives_family_component_and_unit(setup):
    rows = mapping_template(setup.review)["mappings"]
    for row, name in zip(rows, ["M1", "M2", "S1"]):
        row["control_name"] = name
    ring = [SimpleNamespace(FamName=name) for name in ("Q1", "Q2", "SQ1")]
    resolved = validate_mapping_entries(setup.review, setup.connection, rows, ring)
    assert all(row["status"] == "mapped" for row in resolved)


def test_explicit_ordinal_maps_when_fit_display_name_differs(setup):
    setup.review.items[0].name = "FIT display alias"
    explicit = [{
        "correction_type": "normal_quadrupole", "lattice_name": "Q1",
        "lattice_ordinal": 0, "control_name": "M1",
        "component": "B2L", "unit": "m^-1",
    }]
    rows = resolve_mapping(setup.review, setup.connection.inventory, explicit)
    assert rows[0]["status"] == "mapped"
    assert rows[0]["magnet_name"] == "M1"


def test_minimal_template_ordinal_derives_lattice_name_for_correct(setup, monkeypatch):
    setup.review.items[0].name = "FIT display alias"
    document = {
        "schema": "pyloco.machine_mapping", "schema_version": 3,
        "template": True,
        "mappings": [{"pyaml_device": "M1", "role": "normal_quadrupole",
                      "lattice_ordinal": 0}],
    }
    ring = [SimpleNamespace(FamName="Q1", Device="device/Q1")]
    monkeypatch.setattr(
        "pyLOCO.correct.pyaml_connection.lattice_identity",
        lambda _path: ({"sha256": "unused", "element_count": 1}, ring),
    )
    selected, rows, provenance, _identity = validate_mapping_document(
        setup.review, setup.connection, document, "unused.mat"
    )
    assert selected[0]["lattice_name"] == "Q1"
    assert rows[0]["status"] == "mapped"
    assert rows[0]["magnet_name"] == "M1"
    assert "minimal template" in provenance


def test_embedded_machine_mapping_requires_matching_connection_fingerprints(setup, monkeypatch):
    from pyLOCO.machine_mapping import entry
    rows=[]
    for item,name in zip(setup.review.items,("M1","M2","S1")):
        rows.append(entry(role=item.correction_type,pyaml_device=name,
            lattice_ordinal=item.lattice_ordinal,lattice_fam_name=item.name,
            lattice_device=f"device/{item.name}",component="B2L" if item.correction_type=="normal_quadrupole" else "A2L",unit="m^-1"))
    from pyLOCO.correct.pyaml_connection import inventory_fingerprint
    document={"schema":"pyloco.machine_mapping","schema_version":3,
        "reference_lattice":{"sha256":"abc","element_count":3},
        "machine_profile":{"configuration_sha256":setup.connection.config_hash,"mode":"live",
                           "normal_array":"Quads","skew_array":"SkewQuads"},
        "inventories":{"correction":{"sha256":inventory_fingerprint(setup.connection.inventory),
                                      "device_count":3}},"mappings":rows}
    ring=[SimpleNamespace(FamName=item.name,Device=f"device/{item.name}") for item in setup.review.items]
    monkeypatch.setattr("pyLOCO.correct.pyaml_connection.lattice_identity",
                        lambda _path: ({"sha256":"abc","element_count":3},ring))
    selected,resolved,provenance,_identity=validate_mapping_document(
        setup.review,setup.connection,document,"ring.mat")
    assert len(selected)==3 and all(row["status"]=="mapped" for row in resolved)
    assert provenance=="revalidated saved mapping"
    document["machine_profile"]["configuration_sha256"]="stale"
    with pytest.raises(ValueError,match="different machine configuration"):
        validate_mapping_document(setup.review,setup.connection,document,"ring.mat")
    selected,resolved,provenance,_identity=validate_mapping_document(
        setup.review,setup.connection,document,"ring.mat",operator_selected=True)
    assert len(selected)==3 and all(row["status"]=="mapped" for row in resolved)
    assert "explicitly selected" in provenance


def test_minimal_shared_template_accepts_ordinals_and_validates_connected_targets(setup,monkeypatch):
    document={"schema":"pyloco.machine_mapping","schema_version":3,"template":True,
        "mappings":[
            {"pyaml_device":name,"role":item.correction_type,
             "lattice_ordinal":item.lattice_ordinal}
            for item,name in zip(setup.review.items,("M1","M2","S1"))]}
    ring=[SimpleNamespace(FamName=item.name,Device="unrelated/device")
          for item in setup.review.items]
    monkeypatch.setattr("pyLOCO.correct.pyaml_connection.lattice_identity",
                        lambda _path: ({"sha256":"active","element_count":3},ring))
    selected,resolved,provenance,_identity=validate_mapping_document(
        setup.review,setup.connection,document,"ring.mat")
    assert len(selected)==3
    assert all(row["status"]=="mapped" for row in resolved)
    assert "minimal template" in provenance
    document["mappings"][0]["pyaml_device"]="not-configured"
    with pytest.raises(ValueError,match="Unknown or wrong-family mapping target"):
        validate_mapping_document(setup.review,setup.connection,document,"ring.mat")


def test_ebs_smoke_identity_mapping_is_explicit_unique_and_family_checked(tmp_path):
    source = tmp_path / "ebs-smoke.json"; source.write_text("{}")
    identities = [("QD3J", 27, "srmag/m-qd3/c04-a-B2L"),
                  ("QF4J", 36, "srmag/m-qf4/c04-a-B2L"),
                  ("QD5B", 51, "srmag/m-qd5/c04-b-B2L")]
    review = CorrectionReview([CorrectItem(
        index, "normal_quadrupole", lattice, ordinal, "m^-2", 1.0, .9, -.1, .1,
        "test", metadata={"magnetic_length_m": .2})
        for index, (lattice, ordinal, _device) in enumerate(identities)], str(source))
    inventory = [dict(name=device, family="normal_quadrupole", lattice_name=device)
                 for _lattice, _ordinal, device in identities]
    explicit = [dict(lattice_name=lattice, lattice_ordinal=ordinal,
                     control_name=device, component="B2L", unit="m^-1")
                for lattice, ordinal, device in identities]
    rows = resolve_mapping(review, inventory, explicit)
    assert [(row["lattice_name"], row["lattice_ordinal"], row["magnet_name"], row["status"])
            for row in rows] == [(lattice, ordinal, device, "mapped")
                                for lattice, ordinal, device in identities]
    with pytest.raises(ValueError, match="Duplicate mapping target"):
        resolve_mapping(review, inventory, [explicit[0], {**explicit[1], "control_name": explicit[0]["control_name"]}])
    with pytest.raises(ValueError, match="wrong-family"):
        resolve_mapping(review, [{**inventory[0], "family": "skew_quadrupole"}, *inventory[1:]], explicit)
    with pytest.raises(ValueError, match="requires integrated B2L"):
        resolve_mapping(review, inventory, [{**explicit[0], "component": "B2"}, *explicit[1:]])
    with pytest.raises(ValueError, match=r"require m\^-1"):
        resolve_mapping(review, inventory, [{**explicit[0], "unit": "m^-2"}, *explicit[1:]])


def test_integrated_control_values_keep_k_and_kl_distinct(setup):
    item = setup.review.items[0]
    item.metadata.update(control_component="B2L", control_unit="m^-1")
    item.machine_value = 2.0  # pyAML B2L readback: KL [m^-1]
    item.global_scale = .25
    assert item.magnetic_length_m == pytest.approx(.5)
    assert item.current_integrated_value == pytest.approx(2.0)
    assert item.current_gradient_value == pytest.approx(4.0)
    assert item.final_delta == pytest.approx(.05)
    assert item.integrated_final_delta == pytest.approx(.025)
    assert item.applied_control_delta == pytest.approx(.025)
    assert item.target_value == pytest.approx(4.05)
    assert item.target_integrated_value == pytest.approx(2.025)
    assert item.applied_control_unit == "m⁻¹"


def test_preview_integrated_values_and_cumulative_restore(setup):
    setup.review.items[0].individual_scale = .5
    tx = setup.transaction
    first = tx.preview(setup.review)
    assert [row["target"] for row in first["changes"]] == pytest.approx([1.005, 1.996, 3.006])
    assert all(not magnet.strength.writes for magnet in setup.magnets.values())
    with pytest.raises(PermissionError):
        tx.apply(setup.review)
    for count in range(1, 11):
        if count > 1:
            tx.preview(setup.review)
        tx.apply(setup.review, confirmed=True)
        assert setup.magnets["M1"].strength.value == pytest.approx(1 + count * .005)
    with pytest.raises(ValueError, match="100%"):
        tx.preview(setup.review)
    tx.restore()
    assert [magnet.strength.value for magnet in setup.magnets.values()] == [1, 2, 3]
    assert not tx.pending


@pytest.mark.parametrize("length", [None, 0, -1, float("nan")])
def test_invalid_length_blocks(setup, length):
    setup.review.items[0].metadata["magnetic_length_m"] = length
    with pytest.raises(ValueError, match="length"):
        setup.transaction.preview(setup.review)


def test_excluded_unmatched_and_extra_inventory(setup):
    setup.review.items[0].name = "unknown"
    with pytest.raises(ValueError, match="Unresolved"):
        setup.transaction.preview(setup.review)
    setup.review.items[0].included = False
    assert len(setup.transaction.preview(setup.review)["changes"]) == 2


@pytest.mark.parametrize("change", ["config", "readback", "scale", "source", "inventory", "excluded"])
def test_stale_preview_blocks_without_writes(setup, change):
    setup.transaction.preview(setup.review)
    if change == "config": setup.config.write_text("changed")
    if change == "source": setup.source.write_text("changed")
    if change == "readback": setup.magnets["M1"].strength.value = 100
    if change == "scale": setup.review.items[0].individual_scale = .3
    if change == "excluded": setup.review.items[0].included = False
    if change == "inventory": setup.machine.magnets["Quads"].members.reverse()
    with pytest.raises(RuntimeError):
        setup.transaction.apply(setup.review, confirmed=True)
    assert all(not magnet.strength.writes for magnet in setup.magnets.values())


def test_write_pending_persisted_and_failed_write_restored(setup):
    tx = setup.transaction; tx.preview(setup.review)
    def after_write():
        record = json.loads(tx.path.read_text())
        assert record["restore_required"]
        assert "M2" in record["attempted"]
    setup.magnets["M2"].strength.after_write = after_write
    setup.magnets["M2"].strength.fail = lambda value: value != 2
    with pytest.raises(RuntimeError, match="original strengths restored"):
        tx.apply(setup.review, confirmed=True)
    assert [magnet.strength.value for magnet in setup.magnets.values()] == [1, 2, 3]
    assert setup.magnets["M2"].strength.writes == pytest.approx([1.996, 2])


def test_cancel_restores(setup):
    cancel = Event()
    setup.magnets["M1"].strength.after_write = cancel.set
    setup.transaction.preview(setup.review)
    with pytest.raises(RuntimeError, match="cancelled"):
        setup.transaction.apply(setup.review, confirmed=True, cancel=cancel)
    assert setup.magnets["M1"].strength.value == 1
    assert not setup.transaction.pending


def test_restore_failure_recovery_and_wrong_identity(setup):
    tx = setup.transaction; tx.preview(setup.review)
    setup.magnets["M2"].strength.fail = lambda value: True
    with pytest.raises(RuntimeError, match="restoration failed"):
        tx.apply(setup.review, confirmed=True)
    assert tx.pending
    assert tx.record["status"] == "restore_failed"
    with pytest.raises(RuntimeError):
        tx.preview(setup.review)
    recovered = PyAMLCorrectionTransaction(setup.connection, tx.directory)
    recovered.recover(tx.path)
    setup.magnets["M2"].strength.fail = None
    recovered.restore()
    assert not recovered.pending
    assert all(magnet.strength.value == index for index, magnet in enumerate(setup.magnets.values(), 1))


def test_recovery_refuses_changed_config(setup):
    tx = setup.transaction; tx.preview(setup.review); tx.apply(setup.review, confirmed=True)
    setup.config.write_text("different")
    recovered = PyAMLCorrectionTransaction(setup.connection, tx.directory)
    with pytest.raises(RuntimeError, match="configuration changed"):
        recovered.recover(tx.path)


def test_adapter_strength_get_mismatch_restores(setup):
    strength = setup.magnets["M1"].strength
    strength.after_write = lambda: setattr(strength, "value", strength.value + .01 if strength.value != 1 else 1)
    setup.transaction.preview(setup.review)
    with pytest.raises(RuntimeError, match=r"Adapter strength\.get\(\) mismatch"):
        setup.transaction.apply(setup.review, confirmed=True)
    assert strength.value == 1


def test_read_only_connection_never_applies(setup):
    setup.connection.allow_writes = False
    setup.transaction.preview(setup.review)
    with pytest.raises(PermissionError):
        setup.transaction.apply(setup.review, confirmed=True)
    assert all(not magnet.strength.writes for magnet in setup.magnets.values())


def test_mapping_file_changes_stale_preview(setup, tmp_path):
    path = tmp_path / "mapping.json"
    path.write_text('{"mappings": []}')
    setup.transaction.mapping_path = path
    setup.transaction.preview(setup.review)
    path.write_text('{"mappings": [], "changed": true}')
    with pytest.raises(RuntimeError, match="Stale preview"):
        setup.transaction.apply(setup.review, confirmed=True)


def test_unrelated_inventory_is_allowed(setup):
    setup.machine.magnet["EXTRA"] = Magnet("UNFITTED", 4)
    setup.machine.magnets["Quads"].members.append("EXTRA")
    setup.connection.inventory = setup.connection.discover()
    assert len(setup.transaction.preview(setup.review)["changes"]) == 3


def test_cancel_before_first_write_is_not_pending(setup):
    setup.transaction.preview(setup.review)
    cancel = Event(); cancel.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        setup.transaction.apply(setup.review, confirmed=True, cancel=cancel)
    assert not setup.transaction.pending
    assert all(not magnet.strength.writes for magnet in setup.magnets.values())


def test_unsupported_family_and_excluded_correction(setup):
    setup.review.items[0].correction_type = "quadrupole_tilt"
    with pytest.raises(ValueError, match="unsupported"):
        setup.transaction.preview(setup.review)
    setup.review.items[0].included = False
    setup.review.set_global_scale(.25)
    assert setup.transaction.preview(setup.review)["fraction"] == .25


@pytest.mark.parametrize("fraction", [.00123, .05, .125, .25, .333, .5, 1.0])
def test_configurable_fraction_integrated_targets(setup, fraction):
    setup.review.set_global_scale(fraction)
    setup.review.items[0].individual_scale = .5
    tx = setup.transaction
    record = tx.preview(setup.review)
    expected = [1 + .05 * fraction, 2 - .04 * fraction, 3 + .06 * fraction]
    assert record["fraction"] == fraction
    assert [row["target"] for row in record["changes"]] == pytest.approx(expected)
    assert [row["applied_delta_k_m2"] for row in record["changes"]] == pytest.approx([.1 * fraction, -.1 * fraction, .3 * fraction])
    assert [row["applied_delta_kl_m1"] for row in record["changes"]] == pytest.approx([.05 * fraction, -.04 * fraction, .06 * fraction])
    assert all(row["unit"] == "m^-1" for row in record["changes"])
    assert all(not magnet.strength.writes for magnet in setup.magnets.values())
    tx.apply(setup.review, confirmed=True)
    assert [magnet.strength.value for magnet in setup.magnets.values()] == pytest.approx(expected)
    if fraction <= .5:
        record = tx.preview(setup.review)
        assert record["fraction"] == pytest.approx(2 * fraction)
        assert [row["target"] for row in record["changes"]] == pytest.approx(
            [1 + .1 * fraction, 2 - .08 * fraction, 3 + .12 * fraction])
        tx.apply(setup.review, confirmed=True)
    tx.restore()
    assert [magnet.strength.value for magnet in setup.magnets.values()] == [1, 2, 3]


@pytest.mark.parametrize("fraction", [0, -.1, 1.01, float("nan"), float("inf")])
def test_invalid_fraction_blocks_preview(setup, fraction):
    setup.review.global_scale = fraction
    with pytest.raises(ValueError, match="fraction"):
        setup.transaction.preview(setup.review)
    assert all(not magnet.strength.writes for magnet in setup.magnets.values())


def test_custom_fraction_cannot_exceed_cumulative_limit(setup):
    setup.review.set_global_scale(.3)
    for _ in range(3):
        setup.transaction.preview(setup.review)
        setup.transaction.apply(setup.review, confirmed=True)
    with pytest.raises(ValueError, match="100%"):
        setup.transaction.preview(setup.review)
    setup.transaction.restore()


def test_duplicate_fit_names_do_not_silently_share_target(setup):
    setup.review.items[1].name = "Q1"
    rows = resolve_mapping(setup.review, setup.connection.inventory)
    assert rows[0]["status"] == rows[1]["status"] == "duplicate"
    with pytest.raises(ValueError, match="duplicate"):
        setup.transaction.preview(setup.review)


def test_readonly_recovery_requires_reconnection(setup):
    tx = setup.transaction; tx.preview(setup.review); tx.apply(setup.review, confirmed=True)
    setup.connection.allow_writes = False
    recovered = PyAMLCorrectionTransaction(setup.connection, tx.directory)
    with pytest.raises(PermissionError, match="writes enabled"):
        recovered.recover(tx.path)
    assert not recovered.pending
