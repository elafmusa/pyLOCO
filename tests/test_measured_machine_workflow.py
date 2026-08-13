from pathlib import Path

import numpy as np
import pytest

from pyLOCO.measured_machine.workflow import (
    _file_spec,
    _remove_configured_correctors,
    select_elements,
)


class Element:
    def __init__(self, common, family):
        self.CommonName = common
        self.FamName = family


class Ring(list):
    pass


def test_file_spec_preserves_new_mapping_and_adapts_legacy_path():
    assert _file_spec("orm.h5", dataset="response_matrix") == {
        "file": "orm.h5", "dataset": "response_matrix"}
    assert _file_spec({"file": "orm.h5", "dataset": "M"})["dataset"] == "M"


def test_ordered_name_selection_and_missing_name(tmp_path):
    ring = Ring([Element("B", "M2"), Element("A", "M1")])
    names = tmp_path / "names.txt"
    names.write_text("A\nB\n", encoding="utf-8")
    selected = select_elements(ring, {"names_file": "names.txt"}, tmp_path, "bpms")
    np.testing.assert_array_equal(selected, [1, 0])
    names.write_text("missing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not found"):
        select_elements(ring, {"names_file": "names.txt"}, tmp_path, "bpms")


def test_explicit_selection_rejects_duplicates_and_bounds(tmp_path):
    ring = Ring([Element("A", "Q"), Element("B", "Q")])
    with pytest.raises(ValueError, match="duplicate"):
        select_elements(ring, {"indices": [0, 0]}, tmp_path, "quadrupoles")
    with pytest.raises(ValueError, match="outside"):
        select_elements(ring, {"indices": [2]}, tmp_path, "quadrupoles")


def test_pattern_and_regex_selection(tmp_path):
    ring = Ring([Element("A", "QF1"), Element("B", "QD2"), Element("C", "BEND")])
    np.testing.assert_array_equal(select_elements(
        ring, {"pattern": "Q*", "name_attribute": "FamName"}, tmp_path, "quads"), [0, 1])
    np.testing.assert_array_equal(select_elements(
        ring, {"regex": "^QD", "name_attribute": "FamName"}, tmp_path, "quads"), [1])


def test_remove_named_horizontal_corrector_and_orm_column():
    ring = Ring([
        Element("PKH_NOR_01", "HCM"),
        Element("PKH_NOR_47", "HCM"),
        Element("PKV_NOR_01", "VCM"),
    ])
    measured_orm = np.arange(12).reshape(3, 4)
    result = _remove_configured_correctors(
        ring,
        measured_orm,
        np.array([0, 1]),
        np.array([2, 2]),
        [np.array([1.0e-4, 1.0e-4]), np.array([1.0e-4, 1.0e-4])],
        [{"plane": "horizontal", "name": "PKH_NOR_47"}],
    )

    orm, hcor, vcor, steps = result
    np.testing.assert_array_equal(orm, np.delete(measured_orm, 1, axis=1))
    np.testing.assert_array_equal(hcor, [0])
    np.testing.assert_array_equal(vcor, [2, 2])
    np.testing.assert_array_equal(steps[0], [1.0e-4])
    np.testing.assert_array_equal(steps[1], [1.0e-4, 1.0e-4])
