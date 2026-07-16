"""Welcome dashboard for the pyLOCO GUI Project page."""

from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QVBoxLayout,
    QWidget,
)


class ProjectDashboard(QWidget):
    """Project landing page for the polished Milestone 1 shell.

    The dashboard intentionally exposes only shell actions. Buttons emit
    signals that the main window maps to placeholder messages until project
    persistence, documentation links, and example loading are implemented in
    later milestones.
    """

    new_project_requested = Signal()
    open_project_requested = Signal()
    documentation_requested = Signal()
    examples_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("projectDashboard")
        self._build_ui()

    def _build_ui(self) -> None:
        """Create the dashboard layout and action cards."""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(36, 32, 36, 32)
        outer.setSpacing(24)

        title = QLabel("Welcome to pyLOCO")
        title.setObjectName("dashboardTitle")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        subtitle = QLabel(
            "Start a guided LOCO workflow, open an existing analysis project, "
            "or explore documentation and examples."
        )
        subtitle.setObjectName("dashboardSubtitle")
        subtitle.setWordWrap(True)

        outer.addWidget(title)
        outer.addWidget(subtitle)

        cards = QGridLayout()
        cards.setHorizontalSpacing(18)
        cards.setVerticalSpacing(18)
        cards.addWidget(
            self._make_card(
                title="New Project",
                description="Create a new offline LOCO project shell.",
                icon=self.style().standardIcon(QStyle.SP_FileIcon),
                button_text="New Project",
                signal=self.new_project_requested,
            ),
            0,
            0,
        )
        cards.addWidget(
            self._make_card(
                title="Open Project",
                description="Open a saved pyLOCO GUI project when persistence is available.",
                icon=self.style().standardIcon(QStyle.SP_DirOpenIcon),
                button_text="Open Project",
                signal=self.open_project_requested,
            ),
            0,
            1,
        )
        cards.addWidget(
            self._make_card(
                title="Documentation",
                description="Review workflow guidance, data requirements, and future help pages.",
                icon=self.style().standardIcon(QStyle.SP_FileDialogInfoView),
                button_text="Open Documentation",
                signal=self.documentation_requested,
            ),
            1,
            0,
        )
        cards.addWidget(
            self._make_card(
                title="Examples",
                description="Browse planned example project templates for machine-independent studies.",
                icon=self.style().standardIcon(QStyle.SP_DirIcon),
                button_text="Browse Examples",
                signal=self.examples_requested,
            ),
            1,
            1,
        )
        cards.setColumnStretch(0, 1)
        cards.setColumnStretch(1, 1)
        outer.addLayout(cards)

        recent = QFrame()
        recent.setObjectName("recentProjectsPanel")
        recent_layout = QVBoxLayout(recent)
        recent_layout.setContentsMargins(20, 18, 20, 18)
        recent_layout.setSpacing(8)

        recent_title = QLabel("Recent Projects")
        recent_title.setObjectName("sectionTitle")
        recent_placeholder = QLabel("No recent projects yet. Project persistence is planned for a later milestone.")
        recent_placeholder.setObjectName("mutedText")
        recent_placeholder.setWordWrap(True)
        recent_layout.addWidget(recent_title)
        recent_layout.addWidget(recent_placeholder)
        outer.addWidget(recent)
        outer.addStretch(1)

    def _make_card(
        self,
        *,
        title: str,
        description: str,
        icon: QIcon,
        button_text: str,
        signal: Signal,
    ) -> QFrame:
        """Build a dashboard action card."""

        card = QFrame()
        card.setObjectName("dashboardCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        header = QHBoxLayout()
        icon_label = QLabel()
        icon_label.setPixmap(icon.pixmap(28, 28))
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        header.addWidget(icon_label)
        header.addWidget(title_label)
        header.addStretch(1)

        description_label = QLabel(description)
        description_label.setObjectName("cardDescription")
        description_label.setWordWrap(True)

        button = QPushButton(button_text)
        button.setIcon(icon)
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(signal.emit)

        layout.addLayout(header)
        layout.addWidget(description_label)
        layout.addStretch(1)
        layout.addWidget(button, alignment=Qt.AlignRight)
        return card
