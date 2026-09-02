from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from pyLOCO.data_schema import SCHEMA_VERSION
from pyLOCO.gui import __version__
from pyLOCO.measure.app import build_application
from pyLOCO.measure.main_window import MeasureMainWindow, default_mock_devices


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or build_application(["pyloco-measure-branding-test"])


def _dialog_text(dialog) -> str:
    return "\n".join(label.text() for label in dialog.findChildren(QLabel))


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_prominent_brand_and_unchanged_clickable_logo(app, theme):
    window = MeasureMainWindow(devices=default_mock_devices(2))
    window.apply_theme(theme); window.show(); app.processEvents()
    assert window.brand_title.text() == "pyLOCO  MEASURE"
    assert window.brand_title.font().pointSizeF() >= 20
    assert window.brand_title.font().weight() >= 700
    assert window.logo_button.cursor().shape() == Qt.PointingHandCursor
    assert window.logo_button.pixmap() is not None and not window.logo_button.pixmap().isNull()

    QTest.mouseClick(window.logo_button, Qt.LeftButton)
    app.processEvents()
    dialog = window._about_dialog
    assert dialog.windowTitle() == "About pyLOCO Measure"
    assert dialog.isVisible()
    dialog.reject()
    window.close()


def test_about_dialog_reuses_canonical_project_information(app):
    window = MeasureMainWindow(devices=default_mock_devices(2))
    dialog = window._build_about_dialog()
    text = _dialog_text(dialog)
    assert "measurement-acquisition companion to pyLOCO" in text
    assert "fits measured accelerator response data" in text
    assert "consumed directly by pyLOCO" in text
    assert f"Installed pyLOCO version {__version__}" in text
    assert f"Measurement schema {SCHEMA_VERSION}" in text
    assert "Contributors: Elaf Musa" in text
    assert "Ahmed El Deeb" in text
    assert "Scientific reference / methodology" in text
    actions = {button.text() for button in dialog.findChildren(QPushButton)}
    assert {"Documentation", "Methodology", "Copy citation", "Copy BibTeX",
            "Source code", "Report issue"}.issubset(actions)
    assert "WEP5011" in window._software_citation()
    assert "@inproceedings" in window._software_bibtex()
    dialog.close(); window.close()


def test_measurement_help_is_compact_and_context_sensitive(app):
    window = MeasureMainWindow(devices=default_mock_devices(2))
    assert window.measurement_help_group.isCheckable()
    assert not window.measurement_help_group.isChecked()
    assert window.measurement_help_body.isHidden()
    assert window.measurement_help_title.text() == "BPM Noise"
    assert "read-only" in window.measurement_help_text.text()
    assert "measurement weights in pyLOCO" in window.measurement_help_text.text()

    window.measurement_help_group.setChecked(True)
    assert not window.measurement_help_body.isHidden()
    window.measurement_type.setCurrentIndex(window.measurement_type.findData("dispersion"))
    assert window.measurement_help_title.text() == "Dispersion / RF response"
    assert "raw RF-state orbit measurements" in window.measurement_help_text.text()
    assert "used directly by pyLOCO" in window.measurement_help_text.text()
    assert "mean(−Δf) − mean(+Δf)" in window.measurement_help_convention.text()
    assert "canonical signed RF step = f− − f+ = −2Δf" in window.measurement_help_convention.text()
    assert "normalized lattice dispersion" not in (
        window.measurement_help_text.text() + window.measurement_help_convention.text()
    ).lower()

    window.dispersion_direction.setCurrentIndex(window.dispersion_direction.findData("positive"))
    assert "mean(+Δf) − mean(reference)" in window.measurement_help_convention.text()
    assert "canonical RF step = +Δf" in window.measurement_help_convention.text()
    window.dispersion_direction.setCurrentIndex(window.dispersion_direction.findData("negative"))
    assert "mean(−Δf) − mean(reference)" in window.measurement_help_convention.text()
    assert "canonical RF step = −Δf" in window.measurement_help_convention.text()
    window.close()
