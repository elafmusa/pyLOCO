"""Companion-window access to the existing FIT suite appearance preference."""
from PySide6.QtCore import QSettings
from .themes import apply_application_theme, theme_for_key


def suite_appearance_settings():
    # FIT's existing QSettings namespace, independent of companion app names.
    return QSettings("pyLOCO", "pyLOCO GUI")


def ensure_suite_appearance(app):
    active = app.property("pyLOCOTheme")
    if not active:
        active = suite_appearance_settings().value("appearance/theme", "dark")
        apply_application_theme(app, theme_for_key(active))
    return theme_for_key(active)


def select_suite_appearance(app, key):
    theme = theme_for_key(key)
    # An embedded companion shares the FIT QApplication. Use FIT's existing
    # appearance action so its menu, current_theme and plots stay synchronized.
    owner = next((w for w in app.topLevelWidgets()
                  if callable(getattr(w, "_apply_theme_selection", None))), None)
    if owner is not None:
        owner._apply_theme_selection(theme.key)
    else:
        apply_application_theme(app, theme)
    settings = suite_appearance_settings()
    settings.setValue("appearance/theme", theme.key)
    settings.sync()
    return theme
