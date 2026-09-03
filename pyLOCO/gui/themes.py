"""Theme definitions and helpers for the pyLOCO Qt GUI."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import QApplication, QAbstractItemView


@dataclass(frozen=True)
class GuiTheme:
    """A named GUI theme with a Qt stylesheet and plotting colors."""

    key: str
    display_name: str
    stylesheet: str
    plot_face: str
    plot_axes: str
    plot_text: str
    plot_grid: str
    plot_spine: str
    plot_colormap: str = "viridis"


_COMMON_QSS = f"""
* {{ font-size: 12pt; }}
QToolBar#mainToolbar {{ spacing: 10px; padding: 8px 14px; }}
QMainWindow::separator {{ width: 14px; height: 14px; }}
QMainWindow::separator:hover {{ background: #7E57C2; }}
QToolBar#matplotlibToolbar {{ spacing: 2px; padding: 2px; border: 0; }}
QToolBar#matplotlibToolbar QToolButton {{ min-height: 22px; min-width: 22px; padding: 2px; border-radius: 4px; }}
QToolButton, QPushButton {{ border-radius: 6px; font-size: 12pt; font-weight: 600; min-height: 32px; padding: 7px 13px; }}
QToolButton:pressed, QPushButton:pressed, QToolButton:checked {{ padding-top: 9px; padding-left: 17px; }}
QTabBar::tab {{ font-size: 12pt; min-height: 28px; padding: 10px 20px; margin-right: 4px; }}
QTabWidget::pane {{ border-radius: 11px; padding-top: 8px; }}
QGroupBox {{ border-radius: 12px; font-weight: 600; margin-top: 18px; padding: 18px 16px 16px 16px; }}
QGroupBox::title {{ font-size: 13pt; font-weight: 700; subcontrol-origin: margin; left: 14px; padding: 0 8px; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ border-radius: 6px; min-height: 30px; padding: 7px 10px; }}
QComboBox::drop-down {{ border: 0; width: 28px; }}
QTextEdit, QListWidget, QTreeView, QTableView, QTreeWidget#projectExplorerTree {{ border-radius: 9px; padding: 4px; }}
QTreeView::item, QTreeWidget::item, QListWidget::item {{ min-height: 28px; padding: 4px 8px; }}
QTableView::item {{ padding: 7px 8px; }}
QHeaderView::section {{ padding: 8px 10px; font-weight: 700; }}
QLabel#pageTitle {{ font-size: 23pt; font-weight: 750; }}
QLabel#placeholderTitle {{ font-size: 24pt; font-weight: 750; }}
QLabel#dashboardCardTitle {{ font-size: 13pt; font-weight: 700; }}
QLabel#resultsTitle {{ font-size: 20pt; font-weight: 750; }}
QLabel#resultsStatus {{ font-size: 12pt; font-weight: 700; padding: 6px 10px; }}
QLabel#resultMetricTitle {{ font-size: 10pt; font-weight: 600; }}
QLabel#resultMetricValue {{ font-size: 16pt; font-weight: 700; }}
QLabel#placeholderDescription, QLabel#dashboardCardText {{ font-size: 12pt; }}
QStatusBar, QStatusBar QLabel {{ font-size: 11pt; }}
QCheckBox, QRadioButton {{ spacing: 9px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 17px; height: 17px; }}
QProgressBar {{ border-radius: 8px; min-height: 22px; text-align: center; }}
QProgressBar::chunk {{ border-radius: 8px; }}
QScrollBar:vertical {{ width: 13px; }}
QScrollBar:horizontal {{ height: 13px; }}
"""

DARK_THEME = GuiTheme(
    key="dark",
    display_name="Dark",
    plot_face="#2B2D42",
    plot_axes="#34374E",
    plot_text="#F5F5F5",
    plot_grid="#8C92A8",
    plot_spine="#5A6078",
    stylesheet=_COMMON_QSS + """
QMainWindow, QDialog, QWidget { background: #2B2D42; color: #F5F5F5; }
QMenuBar, QMenu, QToolBar#mainToolbar, QStatusBar { background: #2F3247; color: #F5F5F5; border: 0; }
QMenuBar::item { padding: 6px 10px; }
QMenuBar::item:selected, QMenu::item:selected { background: #3C415C; color: #FFFFFF; }
QMenu { border: 1px solid #4B506B; padding: 6px; }
QToolBar#mainToolbar { border-bottom: 1px solid #454A66; }
QToolButton, QPushButton { background: #3C415C; border: 1px solid #59607A; color: #F5F5F5; }
QToolButton:hover, QPushButton:hover { background: #464B68; border-color: #9575CD; }
QToolButton:pressed, QPushButton:pressed, QToolButton:checked { background: #7E57C2; border-color: #B39DDB; color: #FFFFFF; }
QPushButton:disabled, QToolButton:disabled { background: #34374E; color: #8C92A8; border-color: #454A66; }
QToolButton#primaryToolbarAction { background: #7E57C2; border-color: #B39DDB; color: #FFFFFF; font-weight: 700; }
QToolButton#primaryToolbarAction:hover { background: #6B46B1; }
QToolButton#primaryToolbarAction:disabled { background: #34374E; color: #8C92A8; border-color: #454A66; }
QTabWidget::pane { background: #2B2D42; border: 1px solid #454A66; }
QTabBar::tab { background: #34374E; border: 1px solid #454A66; border-bottom: 0; color: #D0D4E0; }
QTabBar::tab:hover { background: #3C415C; }
QTabBar::tab:selected { background: #3C415C; color: #FFFFFF; border-color: #7E57C2; font-weight: 700; }
QDockWidget::title { background: #34374E; color: #FFFFFF; font-weight: 700; padding: 9px 12px; border-bottom: 1px solid #7E57C2; }
QTreeWidget#projectExplorerTree, QTreeView, QTableView, QListWidget, QTextEdit { background: #34374E; alternate-background-color: #3C415C; border: 1px solid #4B506B; color: #F5F5F5; selection-background-color: #7E57C2; selection-color: #FFFFFF; }
QHeaderView::section { background: #3C415C; color: #F5F5F5; border: 0; border-right: 1px solid #4B506B; }
QGroupBox, QWidget#placeholderPageCard, QWidget#dashboardCard { background: #34374E; border: 1px solid #4B506B; color: #F5F5F5; }
QGroupBox::title, QLabel#dashboardCardTitle { color: #D7C6FF; }
QLabel { color: #F5F5F5; }
QLabel#statusPill { background: #3C415C; border: 1px solid #7E57C2; border-radius: 11px; color: #EFE7FF; font-weight: 700; padding: 5px 12px; }
QLabel#placeholderDescription, QLabel#dashboardCardText { color: #D0D4E0; }
QLabel#validationOk { color: #6EE7B7; font-weight: 700; }
QLabel#validationMissing { color: #FBBF24; font-weight: 700; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #3C415C; color: #F5F5F5; border: 1px solid #59607A; selection-background-color: #7E57C2; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #B39DDB; }
QComboBox QAbstractItemView { background: #34374E; color: #F5F5F5; border: 1px solid #7E57C2; selection-background-color: #7E57C2; }
QCheckBox, QRadioButton { color: #F5F5F5; }
QCheckBox::indicator, QRadioButton::indicator { border: 1px solid #8C92A8; background: #3C415C; }
QCheckBox::indicator { border-radius: 4px; }
QRadioButton::indicator { border-radius: 9px; }
QCheckBox::indicator:checked, QRadioButton::indicator:checked { background: #7E57C2; border-color: #B39DDB; }
QProgressBar { background: #3C415C; color: #FFFFFF; border: 1px solid #4B506B; }
QProgressBar::chunk { background: #7E57C2; }
QScrollArea, QScrollBar:vertical, QScrollBar:horizontal { background: #2B2D42; border: 0; }
QScrollBar::handle { background: #59607A; border-radius: 6px; }
QScrollBar::handle:hover { background: #9575CD; }
""",
)

LIGHT_THEME = GuiTheme(
    key="light",
    display_name="Light",
    plot_face="#FFFFFF",
    plot_axes="#FFFFFF",
    plot_text="#222436",
    plot_grid="#C9CEDA",
    plot_spine="#8D95A8",
    stylesheet=_COMMON_QSS + """
QMainWindow, QDialog, QWidget { background: #FFFFFF; color: #222436; }
QMenuBar, QMenu, QToolBar#mainToolbar, QStatusBar { background: #F3F4F8; color: #222436; border: 0; }
QMenuBar::item { padding: 6px 10px; }
QMenuBar::item:selected, QMenu::item:selected { background: #ECE7F8; color: #2D2450; }
QMenu { border: 1px solid #D8DCE8; padding: 6px; }
QToolBar#mainToolbar { border-bottom: 1px solid #D8DCE8; }
QToolButton, QPushButton { background: #FFFFFF; border: 1px solid #CBD1DF; color: #222436; }
QToolButton:hover, QPushButton:hover { background: #F6F2FD; border-color: #9575CD; }
QToolButton:pressed, QPushButton:pressed, QToolButton:checked { background: #7E57C2; border-color: #6B46B1; color: #FFFFFF; }
QPushButton:disabled, QToolButton:disabled { background: #EEF0F5; color: #8C92A8; border-color: #D8DCE8; }
QToolButton#primaryToolbarAction { background: #7E57C2; border-color: #6B46B1; color: #FFFFFF; font-weight: 700; }
QToolButton#primaryToolbarAction:hover { background: #6B46B1; }
QToolButton#primaryToolbarAction:disabled { background: #EEF0F5; color: #8C92A8; border-color: #D8DCE8; }
QTabWidget::pane { background: #FFFFFF; border: 1px solid #D8DCE8; }
QTabBar::tab { background: #F3F4F8; border: 1px solid #D8DCE8; border-bottom: 0; color: #4D5568; }
QTabBar::tab:hover { background: #F8F5FE; }
QTabBar::tab:selected { background: #FFFFFF; color: #2D2450; border-color: #7E57C2; font-weight: 700; }
QDockWidget::title { background: #F3F4F8; color: #222436; font-weight: 700; padding: 9px 12px; border-bottom: 1px solid #7E57C2; }
QTreeWidget#projectExplorerTree, QTreeView, QTableView, QListWidget, QTextEdit { background: #FFFFFF; alternate-background-color: #F7F8FB; border: 1px solid #D8DCE8; color: #222436; selection-background-color: #D7C6FF; selection-color: #1D1733; }
QHeaderView::section { background: #F3F4F8; color: #222436; border: 0; border-right: 1px solid #D8DCE8; }
QGroupBox, QWidget#placeholderPageCard, QWidget#dashboardCard { background: #F7F8FB; border: 1px solid #D8DCE8; color: #222436; }
QGroupBox::title, QLabel#dashboardCardTitle { color: #5E3EA1; }
QLabel { color: #222436; }
QLabel#statusPill { background: #F0EAFB; border: 1px solid #B39DDB; border-radius: 11px; color: #4B2E83; font-weight: 700; padding: 5px 12px; }
QLabel#placeholderDescription, QLabel#dashboardCardText { color: #5F687A; }
QLabel#validationOk { color: #047857; font-weight: 700; }
QLabel#validationMissing { color: #B45309; font-weight: 700; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #FFFFFF; color: #222436; border: 1px solid #CBD1DF; selection-background-color: #D7C6FF; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #7E57C2; }
QComboBox QAbstractItemView { background: #FFFFFF; color: #222436; border: 1px solid #7E57C2; selection-background-color: #D7C6FF; }
QCheckBox, QRadioButton { color: #222436; }
QCheckBox::indicator, QRadioButton::indicator { border: 1px solid #A7AFBF; background: #FFFFFF; }
QCheckBox::indicator { border-radius: 4px; }
QRadioButton::indicator { border-radius: 9px; }
QCheckBox::indicator:checked, QRadioButton::indicator:checked { background: #7E57C2; border-color: #6B46B1; }
QProgressBar { background: #EEF0F5; color: #222436; border: 1px solid #D8DCE8; }
QProgressBar::chunk { background: #7E57C2; }
QScrollArea, QScrollBar:vertical, QScrollBar:horizontal { background: #FFFFFF; border: 0; }
QScrollBar::handle { background: #C3C9D6; border-radius: 6px; }
QScrollBar::handle:hover { background: #9575CD; }
""",
)

THEMES = {theme.key: theme for theme in (LIGHT_THEME, DARK_THEME)}
DEFAULT_THEME_KEY = DARK_THEME.key


def theme_for_key(key: str | None) -> GuiTheme:
    """Return a theme, falling back to the default for unknown saved values."""

    return THEMES.get((key or "").lower(), THEMES[DEFAULT_THEME_KEY])


def apply_application_theme(app: QApplication, theme: GuiTheme) -> None:
    """Apply application-wide typography and visual theme settings."""

    # Keep Qt's native platform font. Requesting optional or generic families
    # makes Qt rebuild fallback aliases for every Suite window on macOS.
    app.setStyleSheet(theme.stylesheet)
    app.setProperty("pyLOCOTheme", theme.key)
    app.setProperty("pyLOCOThemePlot", {
        "face": theme.plot_face,
        "axes": theme.plot_axes,
        "text": theme.plot_text,
        "grid": theme.plot_grid,
        "spine": theme.plot_spine,
        "colormap": theme.plot_colormap,
    })


def configure_item_view(view: QAbstractItemView) -> None:
    """Give Qt item views the room expected by the theme."""

    view.setAlternatingRowColors(True)
    if hasattr(view, "setUniformItemSizes"):
        view.setUniformItemSizes(False)
