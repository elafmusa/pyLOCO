from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from pyLOCO.gui.models.project import ProjectMetadata
from pyLOCO.gui.measurement_mapping import (
    find_exact_device_mappings, inventory_fingerprint, mapping_counts,
    mapping_template, ordered_ordinals, validate_mapping,
)


class BPM:
    def __init__(self, name, device): self.FamName, self.Device = name, device


class Multipole:
    def __init__(self, name, device):
        self.FamName, self.Device = name, device
        self.PolynomA, self.PolynomB = [0.0, 0.0], [0.0, 0.0]
        self.Length, self.PassMethod = 1.0, "StrMPoleSymplectic4Pass"


@pytest.fixture
def example(tmp_path):
    lattice = tmp_path / "ring.mat"; lattice.write_bytes(b"exact lattice")
    config = tmp_path / "machine.yaml"; config.write_text("machine: example\n")
    ring = [BPM("BPM1", "machine/bpm/1"), BPM("BPM2", "machine/bpm/2"),
            Multipole("S1", "machine/steerer/1"), Multipole("S2", "machine/steerer/2")]
    inventory = {
        "bpm": ["machine/bpm/1", "machine/bpm/2"],
        "horizontal_corrector": ["machine/steerer/2-B1L"],
        "vertical_corrector": ["machine/steerer/1-A1L"],
    }
    return ring, inventory, lattice, config


def test_exact_mapping_preserves_measurement_order_and_roles(example):
    ring, inventory, lattice, config = example
    result = find_exact_device_mappings(ring, inventory, lattice_path=lattice, configuration_path=config)
    assert mapping_counts(result) == {
        "mapped": 4, "unresolved": 0, "ambiguous": 0, "duplicate": 0, "invalid": 0, "total": 4,
    }
    assert ordered_ordinals(result, "bpm") == [0, 1]
    assert ordered_ordinals(result, "horizontal_corrector") == [3]
    assert ordered_ordinals(result, "vertical_corrector") == [2]
    rows = {row["role"]: row for row in result["mappings"] if row["role"] != "bpm"}
    assert rows["horizontal_corrector"]["component"] == "B1L"
    assert rows["vertical_corrector"]["component"] == "A1L"


def test_revalidation_rejects_order_inventory_lattice_and_configuration_changes(example, tmp_path):
    ring, inventory, lattice, config = example
    result = find_exact_device_mappings(ring, inventory, lattice_path=lattice, configuration_path=config)
    reversed_inventory = deepcopy(inventory); reversed_inventory["bpm"].reverse()
    with pytest.raises(ValueError, match="inventory differs"):
        validate_mapping(result, ring, reversed_inventory, lattice_path=lattice, configuration_path=config)
    reordered_correctors = deepcopy(inventory)
    reordered_correctors["horizontal_corrector"], reordered_correctors["vertical_corrector"] = (
        reordered_correctors["vertical_corrector"], reordered_correctors["horizontal_corrector"]
    )
    with pytest.raises(ValueError, match="inventory differs"):
        validate_mapping(result, ring, reordered_correctors, lattice_path=lattice, configuration_path=config)
    config.write_text("machine: changed\n")
    with pytest.raises(ValueError, match="different machine/control-system configuration"):
        validate_mapping(result, ring, inventory, lattice_path=lattice, configuration_path=config)
    config.write_text("machine: example\n")
    lattice.write_bytes(b"changed")
    with pytest.raises(ValueError, match="different reference lattice"):
        validate_mapping(result, ring, inventory, lattice_path=lattice, configuration_path=config)


def test_duplicate_wrong_plane_and_unresolved_are_not_validated(example):
    ring, inventory, lattice, config = example
    template = mapping_template(inventory)
    ordinals = [0, 1, 3, 2]
    for row, ordinal in zip(template["mappings"], ordinals): row["lattice_ordinal"] = ordinal
    validated = validate_mapping(template, ring, inventory, lattice_path=lattice,
                                 configuration_path=config, require_provenance=False)
    assert mapping_counts(validated)["mapped"] == 4
    wrong = deepcopy(template); wrong["mappings"][2]["lattice_ordinal"] = 2
    checked = validate_mapping(wrong, ring, inventory, lattice_path=lattice,
                               configuration_path=config, require_provenance=False)
    assert [row for row in checked["mappings"] if row["role"] == "horizontal_corrector"][0]["status"] == "invalid"
    missing = deepcopy(template); missing["mappings"][0]["lattice_ordinal"] = None
    checked = validate_mapping(missing, ring, inventory, lattice_path=lattice,
                               configuration_path=config, require_provenance=False)
    assert mapping_counts(checked)["unresolved"] == 1


def test_saved_status_is_never_trusted(example):
    ring, inventory, lattice, config = example
    template = mapping_template(inventory)
    for row in template["mappings"]:
        row["mapping_status"] = "validated"
    checked = validate_mapping(template, ring, inventory, lattice_path=lattice,
                               configuration_path=config, require_provenance=False)
    assert mapping_counts(checked)["mapped"] == 0


def test_exact_device_does_not_validate_non_steering_corrector(example):
    ring, inventory, lattice, config = example
    ring[3] = BPM("not-a-steerer", "machine/steerer/2")
    checked = find_exact_device_mappings(
        ring, inventory, lattice_path=lattice, configuration_path=config,
    )
    horizontal = [row for row in checked["mappings"] if row["role"] == "horizontal_corrector"][0]
    assert horizontal["status"] == "invalid"
    assert "cannot represent B1L" in horizontal["reason"]


def test_mapping_round_trips_with_fit_project(example, tmp_path):
    ring, inventory, lattice, config = example
    mapping = find_exact_device_mappings(
        ring, inventory, lattice_path=lattice, configuration_path=config,
    )
    project = ProjectMetadata(name="mapped project", measurement_device_mapping=mapping)
    restored = ProjectMetadata.load(project.save(tmp_path / "mapped.pyloco.json"))
    assert restored.measurement_device_mapping == mapping
    assert inventory_fingerprint(inventory) == restored.measurement_device_mapping["inventory"]["sha256"]


EBS = Path(os.environ.get("PYLOCO_EBS_TEST_ROOT", "/external/ebs-pysc-pyaml"))


@pytest.mark.skipif(not all((EBS / name).is_file() for name in ("betamodel.mat", "betamodel_conf_pyaml.yaml")),
                    reason="Validated local EBS artifacts unavailable")
def test_full_ebs_pyaml_inventory_maps_by_exact_device_identity():
    at = pytest.importorskip("at")
    yaml = pytest.importorskip("yaml")
    config_path = EBS / "betamodel_conf_pyaml.yaml"
    config = yaml.safe_load(config_path.read_text())
    arrays = {row["name"]: row["elements"] for row in config["arrays"]}
    inventory = {"bpm": arrays["BPM"], "horizontal_corrector": arrays["HCorr"],
                 "vertical_corrector": arrays["VCorr"]}
    lattice_path = EBS / "betamodel.mat"
    ring = at.load_lattice(str(lattice_path), use="betamodel")
    result = find_exact_device_mappings(ring, inventory, lattice_path=lattice_path,
                                        configuration_path=config_path)
    counts = mapping_counts(result)
    assert counts["mapped"] == counts["total"] == 896
    assert [len(ordered_ordinals(result, role)) for role in inventory] == [320, 288, 288]


SMOKE_ROOT = Path(__file__).parents[1] / "measurements"
SMOKE_FILES = {
    "bpm_noise": SMOKE_ROOT / "pyloco-bpm-noise-20261013-115332.h5",
    "dispersion": SMOKE_ROOT / "pyloco-dispersion-20261013-121920.h5",
    "orm": SMOKE_ROOT / "pyloco-orm-20261013-133434.h5",
}


@pytest.mark.skipif(
    not (EBS / "betamodel.mat").is_file() or not all(path.is_file() for path in SMOKE_FILES.values()),
    reason="Validated modern EBS lattice or smoke session unavailable",
)
def test_real_modern_ebs_smoke_inventory_maps_330_devices_in_measurement_order():
    at = pytest.importorskip("at")
    pytest.importorskip("h5py")
    from pyLOCO.gui.fit_workflow import inspect_measurement_files
    from pyLOCO.gui.measurement_mapping import measurement_inventory

    payload = inspect_measurement_files({role: str(path) for role, path in SMOKE_FILES.items()}, {})
    assert payload["errors"] == []
    inventory = measurement_inventory(payload)
    assert [len(inventory[role]) for role in ("bpm", "horizontal_corrector", "vertical_corrector")] == [320, 5, 5]
    lattice_path = EBS / "betamodel.mat"
    ring = at.load_lattice(str(lattice_path), use="betamodel")
    assert len(ring) == 4434
    result = find_exact_device_mappings(ring, inventory, lattice_path=lattice_path)
    assert mapping_counts(result)["mapped"] == mapping_counts(result)["total"] == 330
    assert [len(ordered_ordinals(result, role)) for role in inventory] == [320, 5, 5]
    assert (2 * len(ordered_ordinals(result, "bpm")),
            len(ordered_ordinals(result, "horizontal_corrector"))
            + len(ordered_ordinals(result, "vertical_corrector"))) == (640, 10)
