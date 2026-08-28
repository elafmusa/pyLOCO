"""Failure-safe access to packaged pyLOCO branding assets."""

from __future__ import annotations

from importlib.resources import files
import logging

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QImage, QPainter, QPixmap, QRegion
from PySide6.QtWidgets import QLabel


_ASSET_PACKAGE = "pyLOCO.gui.assets"
ICON_ASSET = "pyloco_app_icon.png"
RING_ASSET = "pyloco_ring.png"
MASTER_ASSET = "pyloco_logo_master.png"
DISPLAY_ASSET = "pyloco_logo_refined.png"
_THEMED_PIXMAP_CACHE: dict[tuple[str, str], QPixmap] = {}


def asset_bytes(name: str) -> bytes:
    """Return a packaged asset, or empty bytes when branding is unavailable."""

    try:
        return files(_ASSET_PACKAGE).joinpath(name).read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        logging.getLogger(__name__).warning("pyLOCO GUI asset is unavailable: %s", name)
        return b""


def load_pixmap(name: str = ICON_ASSET) -> QPixmap:
    """Load a package resource without assuming a filesystem installation."""

    pixmap = QPixmap()
    data = asset_bytes(name)
    if data:
        pixmap.loadFromData(data)
    return pixmap


def application_icon() -> QIcon:
    """Return the simplified storage-ring application icon."""

    return QIcon(load_pixmap())


def wordmark_colors(theme_key: str = "light") -> tuple[str, str]:
    """Return accessible brand colors without changing the Light artwork."""

    return ("#A98BFF", "#B9D4FF") if theme_key == "dark" else ("#6426D9", "#082B6F")


def wordmark_html(theme_key: str = "light", *, size_px: int = 34) -> str:
    py_color, loco_color = wordmark_colors(theme_key)
    return (
        f'<span style="font-size:{size_px}px;font-weight:700;color:{py_color}">py</span>'
        f'<span style="font-size:{size_px}px;font-weight:700;color:{loco_color}">LOCO</span>'
    )


def wordmark_icon(height: int = 28, theme_key: str = "light") -> QIcon:
    """Render the compact two-color pyLOCO wordmark on transparency."""

    font = QFont("Helvetica Neue", max(12, height - 8), QFont.Bold)
    metrics = QFontMetrics(font)
    py_width = metrics.horizontalAdvance("py")
    loco_width = metrics.horizontalAdvance("LOCO")
    pixmap = QPixmap(py_width + loco_width + 4, height)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.TextAntialiasing)
    painter.setFont(font)
    baseline = (height + metrics.ascent() - metrics.descent()) // 2
    py_color, loco_color = wordmark_colors(theme_key)
    painter.setPen(QColor(py_color))
    painter.drawText(0, baseline, "py")
    painter.setPen(QColor(loco_color))
    painter.drawText(py_width, baseline, "LOCO")
    painter.end()
    return QIcon(pixmap)


def _dark_treated_pixmap(pixmap: QPixmap) -> QPixmap:
    """Retain the selected artwork while making its navy marks visible on dark UI."""

    image = pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            if color.red() > 246 and color.green() > 246 and color.blue() > 246:
                color.setAlpha(0)
            elif color.blue() > color.red() * 1.25 and color.blue() > color.green() * 1.15 and color.lightness() < 95:
                color = QColor("#B9D4FF")
            image.setPixelColor(x, y, color)
    return QPixmap.fromImage(image)


def set_asset(
    label: QLabel, size: QSize, name: str = RING_ASSET, *, crop_transparency: bool = True,
    theme_key: str = "light",
) -> bool:
    """Set a smoothly scaled, aspect-preserving packaged image on ``label``."""

    pixmap = load_pixmap(name)
    if pixmap.isNull():
        label.hide()
        return False
    # Full logos retain the approved artwork in both themes. Their white
    # artwork background provides dark-mode contrast; only the compact toolbar
    # wordmark uses theme-aware text colors.
    bounds = QRegion(pixmap.mask()).boundingRect() if crop_transparency else pixmap.rect()
    visible = pixmap.copy(bounds) if bounds.isValid() else pixmap
    pixel_ratio = label.devicePixelRatioF()
    target = QSize(
        max(1, round(size.width() * pixel_ratio)),
        max(1, round(size.height() * pixel_ratio)),
    )
    rendered = visible.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    rendered.setDevicePixelRatio(pixel_ratio)
    label.setPixmap(rendered)
    label.setFixedSize(size)
    label.setAlignment(Qt.AlignCenter)
    label.setAccessibleName("pyLOCO storage-ring logo")
    label.show()
    return True


def set_logo(label: QLabel, size: QSize) -> bool:
    """Compatibility helper for the exact ring artwork used inside the GUI."""

    return set_asset(label, size, RING_ASSET)
