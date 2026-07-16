"""Placeholder widgets used by the Milestone 1 GUI shell."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget


class DashboardCard(QFrame):
    """Small informational card for non-executing dashboard placeholders."""

    def __init__(self, title: str, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dashboardCard")
        self.setMinimumHeight(118)

        title_label = QLabel(title)
        title_label.setObjectName("dashboardCardTitle")
        title_label.setWordWrap(True)

        text_label = QLabel(text)
        text_label.setObjectName("dashboardCardText")
        text_label.setWordWrap(True)
        text_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        layout.addWidget(title_label)
        layout.addWidget(text_label, 1)


class PlaceholderPage(QWidget):
    """Simple page that documents a planned workflow area.

    Parameters
    ----------
    title:
        Human-readable page title.
    description:
        Short explanation of the functionality planned for a future milestone.
    cards:
        Optional title/text pairs shown as static dashboard cards. These are
        deliberately UI-only and do not import or execute numerical LOCO code.
    """

    def __init__(
        self,
        title: str,
        description: str,
        cards: Sequence[tuple[str, str]] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(f"{title.lower().replace(' ', '_')}PlaceholderPage")

        title_label = QLabel(title)
        title_label.setObjectName("placeholderTitle")
        title_label.setAlignment(Qt.AlignCenter)

        description_label = QLabel(description)
        description_label.setObjectName("placeholderDescription")
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setWordWrap(True)

        card_container = QWidget()
        card_container.setObjectName("placeholderPageCard")
        card_layout = QGridLayout(card_container)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setHorizontalSpacing(16)
        card_layout.setVerticalSpacing(16)

        for index, (card_title, card_text) in enumerate(cards):
            card_layout.addWidget(
                DashboardCard(card_title, card_text), index // 2, index % 2
            )

        if not cards:
            card_layout.addWidget(
                DashboardCard("Planned workspace", "Future controls will appear here."),
                0,
                0,
            )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 32, 36, 36)
        layout.setSpacing(18)
        layout.addStretch(1)
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addSpacing(10)
        layout.addWidget(card_container)
        layout.addStretch(2)
