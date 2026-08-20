"""Non-modal chooser hosting the optional waiting games."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from .game_2048 import Game2048Widget
from .minesweeper import MinesweeperWidget
from .sliding_puzzle import SlidingPuzzleWidget
from .tic_tac_toe import TicTacToeWidget


class WaitingGamesDialog(QDialog):
    STATUS_TEXT = {
        "running": "● LOCO running",
        "completed": "✓ LOCO fit completed",
        "cancelled": "LOCO run cancelled",
        "failed": "LOCO run ended with an error",
    }
    _session_size = None

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("While LOCO runs…")
        self.setModal(False); self.setWindowModality(Qt.NonModal)
        self.resize(self._session_size or self.sizeHint().expandedTo(QSize(430, 500)))
        self.stack = QStackedWidget(); self.chooser = self._chooser(); self.stack.addWidget(self.chooser)
        self.games = {
            "2048": Game2048Widget(), "Minesweeper": MinesweeperWidget(),
            "15 Puzzle": SlidingPuzzleWidget(), "Tic-Tac-Toe": TicTacToeWidget(),
        }
        for widget in self.games.values(): self.stack.addWidget(self._game_page(widget))
        self.loco_status = QLabel(); self.loco_status.setObjectName("statusPill"); self.loco_status.setAlignment(Qt.AlignCenter)
        layout = QVBoxLayout(self); layout.setContentsMargins(18, 16, 18, 14); layout.addWidget(self.stack, 1); layout.addWidget(self.loco_status)
        self.set_loco_status("running")

    def _chooser(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        title = QLabel("While LOCO runs…"); title.setObjectName("gameTitle"); title.setAlignment(Qt.AlignCenter)
        prompt = QLabel("Choose a game"); prompt.setAlignment(Qt.AlignCenter)
        layout.addStretch(1); layout.addWidget(title); layout.addWidget(prompt); layout.addSpacing(8)
        for label, icon in (("2048", "🔢"), ("Minesweeper", "●"), ("15 Puzzle", "🧩"), ("Tic-Tac-Toe", "✕")):
            button = QPushButton(f"{icon}   {label}"); button.setMinimumHeight(42)
            button.clicked.connect(lambda checked=False, name=label: self.show_game(name)); layout.addWidget(button)
        layout.addStretch(1); return page

    def _game_page(self, game: QWidget) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(0, 0, 0, 0)
        back = QPushButton("← Games"); back.setMaximumWidth(105); back.clicked.connect(lambda: self.stack.setCurrentWidget(self.chooser))
        row = QHBoxLayout(); row.addWidget(back); row.addStretch(1); layout.addLayout(row); layout.addWidget(game, 1, Qt.AlignCenter)
        return page

    def show_game(self, name: str) -> None:
        widget = self.games[name]
        for index in range(1, self.stack.count()):
            page = self.stack.widget(index)
            if page.isAncestorOf(widget): self.stack.setCurrentWidget(page); break
        widget.setFocus()

    def set_loco_status(self, state: str) -> None:
        if state not in self.STATUS_TEXT: raise ValueError(f"Unknown LOCO game status: {state}")
        self.loco_status.setText(self.STATUS_TEXT[state]); self.loco_status.setProperty("locoState", state)
        self.loco_status.style().unpolish(self.loco_status); self.loco_status.style().polish(self.loco_status)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        type(self)._session_size = self.size(); super().closeEvent(event)
