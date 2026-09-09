"""Companion-window access to the existing FIT suite appearance preference."""
from PySide6.QtCore import QSettings
from .themes import DEFAULT_ACCENT_KEY, accent_for_key, apply_application_theme, theme_for_key


def suite_appearance_settings():
    # FIT's existing QSettings namespace, independent of companion app names.
    return QSettings("pyLOCO", "pyLOCO GUI")


def ensure_suite_appearance(app):
    active = app.property("pyLOCOTheme")
    if not active:
        active = suite_appearance_settings().value("appearance/theme", "dark")
        accent = suite_appearance_settings().value("appearance/accent", DEFAULT_ACCENT_KEY)
        apply_application_theme(app, theme_for_key(active), accent)
    return theme_for_key(active)


def select_suite_appearance(app, key):
    theme = theme_for_key(key)
    # An embedded companion shares the FIT QApplication. Use FIT's existing
    # appearance action so its menu, current_theme and plots stay synchronized.
    owners = [w for w in app.topLevelWidgets()
              if callable(getattr(w, "_apply_theme_selection", None))]
    if owners:
        for owner in owners: owner._apply_theme_selection(theme.key)
    else:
        apply_application_theme(app, theme)
    settings = suite_appearance_settings()
    settings.setValue("appearance/theme", theme.key)
    settings.sync()
    return theme


def ensure_suite_accent(app):
    key = str(app.property("pyLOCOAccent") or suite_appearance_settings().value("appearance/accent", DEFAULT_ACCENT_KEY))
    return key if key.lower() in {"purple", "blue", "teal", "graphite"} else DEFAULT_ACCENT_KEY


def select_suite_accent(app, key):
    key = key.lower() if key.lower() in {"purple", "blue", "teal", "graphite"} else DEFAULT_ACCENT_KEY
    settings = suite_appearance_settings(); settings.setValue("appearance/accent", key); settings.sync()
    apply_application_theme(app, theme_for_key(app.property("pyLOCOTheme")), key)
    for window in app.topLevelWidgets():
        combo=getattr(window,"accent_combo",None)
        if combo is not None and combo.currentData()!=key:
            combo.blockSignals(True); combo.setCurrentIndex(combo.findData(key)); combo.blockSignals(False)
    return accent_for_key(key)
