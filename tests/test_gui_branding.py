from importlib.resources import files

import pytest


pytest.importorskip("PySide6")

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QApplication, QLabel, QDockWidget

from pyLOCO.gui.app import build_application
from pyLOCO.gui.branding import DISPLAY_ASSET, ICON_ASSET, MASTER_ASSET, application_icon, asset_bytes, load_pixmap, set_logo, wordmark_colors
from pyLOCO.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or build_application(["pyloco-test"])
    yield instance


def test_packaged_icon_can_be_resolved():
    resource = files("pyLOCO.gui.assets").joinpath(ICON_ASSET)
    assert resource.is_file()
    assert asset_bytes(ICON_ASSET).startswith(b"\x89PNG")
    assert asset_bytes(MASTER_ASSET).startswith(b"\x89PNG")
    assert asset_bytes(DISPLAY_ASSET).startswith(b"\x89PNG")


def test_missing_asset_is_failure_safe(app, monkeypatch):
    assert asset_bytes("missing-logo.png") == b""
    assert load_pixmap("missing-logo.png").isNull()
    monkeypatch.setattr("pyLOCO.gui.branding.asset_bytes", lambda _name: b"")
    label = QLabel()
    assert not set_logo(label, QSize(32, 32))
    assert label.isHidden()


@pytest.mark.parametrize("size", [16, 32, 48, 64, 128])
def test_application_icon_can_be_created_at_standard_sizes(app, size):
    icon = application_icon()
    assert not icon.isNull()
    assert not icon.pixmap(size, size).isNull()


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_main_window_initializes_in_each_theme(app, theme):
    window = MainWindow()
    window._apply_theme_selection(theme)
    assert window.current_theme.key == theme
    assert not window.windowIcon().isNull()
    window.close()


@pytest.mark.parametrize(
    ("width", "brand"),
    [(800, False), (900, False), (1200, False), (1500, True)],
)
def test_header_branding_is_responsive(app, width, brand):
    window = MainWindow()
    window.resize(width, 800)
    window._update_header_branding()
    assert window.header_brand_action.isVisible() is brand
    window.close()


def test_header_wordmark_is_an_icon_button_with_scientific_menu(app):
    window = MainWindow()
    wordmark = window.header_brand_label.text()
    assert "font-size:34px" in wordmark
    for color in wordmark_colors(window.current_theme.key): assert color in wordmark
    assert window.header_brand.minimumSizeHint().width() >= 180
    labels = [action.text() for action in window.header_brand.menu().actions()]
    assert "About pyLOCO" in labels
    assert "Copy citation" in labels
    assert "Copy BibTeX" in labels
    assert "Repository / Source code" in labels
    assert "Report an issue" in labels
    window.close()


def test_theme_switch_updates_wordmark_and_full_logo_at_device_pixel_ratio(app):
    window = MainWindow(); window._apply_theme_selection("light")
    light_text = window.header_brand_label.text(); light_key = window.dashboard_logo.pixmap().cacheKey()
    window._apply_theme_selection("dark")
    assert window.header_brand_label.text() != light_text
    assert all(color in window.header_brand_label.text() for color in wordmark_colors("dark"))
    assert window.dashboard_logo.pixmap().cacheKey() != light_key
    assert window.dashboard_logo.pixmap().devicePixelRatio() == pytest.approx(window.dashboard_logo.devicePixelRatioF())
    dialog = window._build_about_dialog()
    logo = next(label for label in dialog.findChildren(QLabel) if label.pixmap())
    assert logo.pixmap().devicePixelRatio() == pytest.approx(logo.devicePixelRatioF())
    dialog.close(); window.close()


def test_dashboard_logo_opens_the_same_scientific_information(app):
    window = MainWindow()
    header_labels = [action.text() for action in window.header_brand.menu().actions()]
    dashboard_labels = [
        action.text() for action in window.dashboard_logo_button.menu().actions()
    ]
    assert dashboard_labels == header_labels
    assert window.dashboard_logo_button.cursor().shape() == Qt.PointingHandCursor
    assert window.dashboard_logo_button.size() == QSize(338, 228)
    logo = window.dashboard_logo_button.findChild(QLabel)
    assert logo is not None
    assert logo.size() == QSize(330, 220)
    assert logo.pixmap() is not None
    assert logo.pixmap().size() == QSize(330, 220)
    window.close()


def test_header_wordmark_is_present_at_the_default_window_width(app):
    window = MainWindow()
    window._update_header_branding()
    assert window.width() >= 900
    assert window.header_brand_action.isVisible()
    window.close()


def test_project_explorer_restores_native_dock_behavior(app):
    window = MainWindow()
    dock = window._project_explorer
    assert dock.minimumWidth() == 0
    assert dock.features() & QDockWidget.DockWidgetClosable
    assert dock.features() & QDockWidget.DockWidgetMovable
    assert dock.features() & QDockWidget.DockWidgetFloatable
    assert dock.toggleViewAction().shortcut().toString()
    window.close()


def test_about_dialog_uses_real_project_metadata(app):
    window = MainWindow()
    dialog = window._build_about_dialog()
    text = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "Apache-2.0" in text
    assert "github.com/elafmusa/pyLOCO" in text
    assert "PyLOCO: A Python Framework for Linear Optics Correction in Storage Rings" in text
    assert "Elaf Musa" in text
    assert "Ahmed El Deeb" in text
    assert "With thanks to" in text
    assert "Simone Liuzzo" in text
    assert "WEP5011" in text
    assert "STORAGE RING OPTICS CORRECTION" in text
    logo_pixmaps = [label.pixmap() for label in dialog.findChildren(QLabel) if label.pixmap()]
    assert any(pixmap.size() == QSize(300, 200) for pixmap in logo_pixmaps)
    dialog.close()
    window.close()


def test_paper_citation_and_bibtex_use_verified_metadata(app):
    window = MainWindow()
    assert "WEP5011" in window._software_citation()
    assert "indico.jacow.org/event/95/contributions/13338" in window._software_citation()
    assert "Musa, Elaf" in window._software_bibtex()
    assert "IPAC'26" in window._software_bibtex()
    window.close()
