from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from pyLOCO.gui.main_window import MainWindow
from pyLOCO.gui.models.project import ProjectMetadata
from pyLOCO.machine_mapping import (CORRECTION_ROLES, correction_projection,
                                    pyaml_configuration_inventory)
from pyLOCO.measure.full_session import save_full_session_bundle
from pyLOCO.measure.main_window import MeasureMainWindow


REPOSITORY_ROOT = Path(__file__).parents[1]
EBS_SESSION = REPOSITORY_ROOT / "measurements/measurement-session-20261023-191438.pyloco-session.json"


def _role_counts(document):
    return Counter(row.get("role") for row in document.get("mappings", ()))


@pytest.mark.skipif(not EBS_SESSION.is_file(), reason="October 23 EBS session unavailable")
def test_ebs_gui_export_session_fit_reopen_and_correct_preserve_complete_mapping(tmp_path, monkeypatch):
    """Exercise the GUI serialization boundaries, not only the generator."""
    yaml = pytest.importorskip("yaml")
    pytest.importorskip("at")
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.Ok)

    source = MeasureMainWindow()
    try:
        source.open_full_measurement_session(EBS_SESSION)
        app.processEvents()
        configuration = Path(source.project.metadata["pyaml_options"]["config_path"])
        configured, _metadata = pyaml_configuration_inventory(
            yaml.safe_load(configuration.read_text(encoding="utf-8"))
        )
        expected = {role: len(names) for role, names in configured.items() if names}
        assert sum(expected.values()) == 1954

        # Reproduce the reported state: the older session contains only its
        # selected 320/5/5 measurement rows before validation upgrades it.
        assert len(source.machine_mapping_document["mappings"]) == 330
        legacy_bundle = save_full_session_bundle(
            tmp_path / "legacy-input.pyloco-measure-session",
            project=source._collect_project(), session_manifest=EBS_SESSION,
            reference_lattice=source.reference_model.path,
            reference_mat_variable=source.reference_model.mat_variable,
            reference_element_count=len(source.reference_model.ring),
            mapping=source.machine_mapping_document,
        )
    finally:
        source.close(); app.processEvents()

    measure = MeasureMainWindow()
    try:
        measure.open_full_measurement_session(legacy_bundle.manifest_path)
        measure._verify_device_alignment()
        app.processEvents()

        exported_mapping = tmp_path / "machine-mapping.json"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(exported_mapping), ""))
        measure._save_validated_mapping()
        exported = json.loads(exported_mapping.read_text(encoding="utf-8"))
        assert len(exported["mappings"]) == sum(expected.values())
        assert _role_counts(exported) == Counter(expected)

        # The ordinary Save action must persist that same master in the full
        # session while keeping the selected ORM inventory separate.
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(legacy_bundle.manifest_path), ""))
        monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)
        assert measure.save_active_session()
        saved_manifest = json.loads(legacy_bundle.manifest_path.read_text(encoding="utf-8"))
        saved_master = saved_manifest["metadata"]["full_measure_session"]["mapping"]
        assert len(saved_master["mappings"]) == sum(expected.values())
        assert _role_counts(saved_master) == Counter(expected)
        assert len(measure.measurement_device_alignment["mappings"]) == 330
    finally:
        measure.close(); app.processEvents()

    fit = MainWindow()
    try:
        monkeypatch.setattr(fit, "_choose_measurement_session_import_mode", lambda *args: "full")
        assert fit.open_measurement_session(legacy_bundle.manifest_path)
        assert len(fit.project.machine_mapping["mappings"]) == sum(expected.values())
        assert _role_counts(fit.project.machine_mapping) == Counter(expected)
        assert len(fit.project.measurement_device_mapping["mappings"]) == 330
        assert fit.machine_mapping_details_button.isEnabled()
        assert fit.mapping_table_scope.currentData() == "machine"
        assert fit.measurement_mapping_table.rowCount() == 1954
        assert fit.mapping_table_scope.currentText() == "Complete Machine Mapping (1,954 configured devices)"
        measurement_scope=fit.mapping_table_scope.findData("measurement")
        assert fit.mapping_table_scope.itemText(measurement_scope) == "Measured BPM/H/V mapping (330 selected devices)"

        # Exercise the actual FIT button slot. It must rediscover/revalidate
        # the complete configured inventory without changing measured order
        # or the independent fitting recipe.
        recipe_before = json.dumps(fit.project.fit_recipe, sort_keys=True)
        measured_before = [
            (row["role"], row["measured_name"], row["lattice_ordinal"])
            for row in fit.project.measurement_device_mapping["mappings"]
        ]
        assert fit.find_measurement_mappings()
        assert len(fit.project.machine_mapping["mappings"]) == sum(expected.values())
        assert _role_counts(fit.project.machine_mapping) == Counter(expected)
        assert sum(1 for row in fit.project.machine_mapping["mappings"]
                   if row["role"] in ("bpm", "horizontal_corrector", "vertical_corrector")) == 896
        assert [(row["role"], row["measured_name"], row["lattice_ordinal"])
                for row in fit.project.measurement_device_mapping["mappings"]] == measured_before
        assert json.dumps(fit.project.fit_recipe, sort_keys=True) == recipe_before
        assert "Machine Mapping: 1,954 configured devices" in fit.measurement_mapping_feedback.text()
        assert "Measurement Mapping: 896 available BPM/H/V devices" in fit.measurement_mapping_feedback.text()
        assert fit.mapping_table_scope.currentData() == "machine"
        assert fit.measurement_mapping_table.rowCount() == 1954
        normal_index = fit.mapping_table_role_filter.findData("normal_quadrupole")
        assert normal_index >= 0
        fit.mapping_table_role_filter.setCurrentIndex(normal_index)
        assert fit.measurement_mapping_table.rowCount() == expected["normal_quadrupole"]
        fit.show_measurement_mapping_details()
        assert fit.mapping_table_scope.currentData() == "measurement"
        assert fit.measurement_mapping_table.rowCount() == 330
        fit.show_complete_machine_mapping_details()
        assert fit.mapping_table_scope.currentData() == "machine"
        assert fit.measurement_mapping_table.rowCount() == 1954

        project_path = tmp_path / "imported-complete-mapping.pyloco.json"
        assert fit._save_project_to(project_path)
    finally:
        fit.project.modified = False
        fit.close(); app.processEvents()

    reopened = ProjectMetadata.load(project_path)
    assert len(reopened.machine_mapping["mappings"]) == sum(expected.values())
    assert _role_counts(reopened.machine_mapping) == Counter(expected)
    correction_rows = correction_projection(reopened.machine_mapping)
    expected_correction = sum(expected.get(role, 0) for role in CORRECTION_ROLES)
    assert len(correction_rows) == expected_correction
    assert all(row["control_name"] and row["lattice_ordinal"] is not None for row in correction_rows)
