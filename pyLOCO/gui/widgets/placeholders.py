"""Placeholder widgets used by the Milestone 1 GUI shell."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlaceholderPage(QWidget):
    """Simple page that documents a planned workflow area.

    Parameters
    ----------
    title:
        Human-readable page title.
    description:
        Short explanation of the functionality planned for a future milestone.
    """

    def __init__(self, title: str, description: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(f"{title.lower().replace(' ', '_')}PlaceholderPage")

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: 600;")

        description_label = QLabel(description)
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setWordWrap(True)
        description_label.setStyleSheet("font-size: 14px; color: #555;")

        layout = QVBoxLayout(self)
        layout.addStretch(1)
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addStretch(2)
