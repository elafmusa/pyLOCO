"""Beginner Minesweeper model and compact Qt view."""

from __future__ import annotations

import random

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class MinesweeperGame:
    def __init__(self, rows: int = 9, columns: int = 9, mines: int = 10,
                 rng: random.Random | None = None) -> None:
        if not 0 < mines < rows * columns:
            raise ValueError("Mine count must fit inside the board")
        self.rows, self.columns, self.mine_count = rows, columns, mines
        self.rng = rng or random.Random(); self.new_game()

    def new_game(self) -> None:
        cells = [(r, c) for r in range(self.rows) for c in range(self.columns)]
        self.mines = set(self.rng.sample(cells, self.mine_count))
        self.revealed: set[tuple[int, int]] = set()
        self.flags: set[tuple[int, int]] = set()
        self.lost = False; self.won = False

    def neighbors(self, row: int, column: int):
        for nr in range(max(0, row - 1), min(self.rows, row + 2)):
            for nc in range(max(0, column - 1), min(self.columns, column + 2)):
                if (nr, nc) != (row, column):
                    yield nr, nc

    def adjacent_mines(self, row: int, column: int) -> int:
        return sum(cell in self.mines for cell in self.neighbors(row, column))

    def toggle_flag(self, row: int, column: int) -> bool:
        cell = (row, column)
        if self.lost or self.won or cell in self.revealed:
            return False
        if cell in self.flags: self.flags.remove(cell)
        else: self.flags.add(cell)
        return True

    def reveal(self, row: int, column: int) -> bool:
        start = (row, column)
        if self.lost or self.won or start in self.flags or start in self.revealed:
            return False
        if start in self.mines:
            self.revealed.add(start); self.lost = True; return True
        pending = [start]
        while pending:
            cell = pending.pop()
            if cell in self.revealed or cell in self.flags or cell in self.mines:
                continue
            self.revealed.add(cell)
            if self.adjacent_mines(*cell) == 0:
                pending.extend(self.neighbors(*cell))
        safe = self.rows * self.columns - self.mine_count
        self.won = len(self.revealed) == safe
        return True


class MineButton(QPushButton):
    right_clicked = Signal()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.RightButton:
            self.right_clicked.emit(); event.accept(); return
        super().mousePressEvent(event)


class MinesweeperWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent); self.game = MinesweeperGame()
        title = QLabel("Minesweeper"); title.setObjectName("gameTitle")
        self.status = QLabel()
        grid = QGridLayout(); grid.setSpacing(2); self.cells = []
        for row in range(self.game.rows):
            line = []
            for column in range(self.game.columns):
                button = MineButton(); button.setFixedSize(34, 34)
                button.clicked.connect(lambda checked=False, r=row, c=column: self._reveal(r, c))
                button.right_clicked.connect(lambda r=row, c=column: self._flag(r, c))
                grid.addWidget(button, row, column); line.append(button)
            self.cells.append(line)
        new_button = QPushButton("New Game"); new_button.clicked.connect(self._new)
        layout = QVBoxLayout(self); layout.addWidget(title); layout.addWidget(self.status); layout.addLayout(grid); layout.addWidget(new_button)
        self.refresh()

    def _new(self) -> None: self.game.new_game(); self.refresh()
    def _reveal(self, row: int, column: int) -> None: self.game.reveal(row, column); self.refresh()
    def _flag(self, row: int, column: int) -> None: self.game.toggle_flag(row, column); self.refresh()

    def refresh(self) -> None:
        remaining = self.game.mine_count - len(self.game.flags)
        message = f"Mines remaining: {remaining}"
        if self.game.won: message = "🎉 All safe cells revealed!"
        elif self.game.lost: message = "Mine revealed — try again"
        self.status.setText(message)
        for row in range(self.game.rows):
            for column in range(self.game.columns):
                cell = (row, column); button = self.cells[row][column]
                if cell in self.game.revealed:
                    if cell in self.game.mines: text, color = "●", "#b94a48"
                    else:
                        count = self.game.adjacent_mines(row, column); text, color = (str(count) if count else ""), "#dfe3ec"
                    button.setText(text); button.setStyleSheet(f"background:{color}; color:#252638; border-radius:4px;")
                elif self.game.lost and cell in self.game.mines:
                    button.setText("●"); button.setStyleSheet("background:#b94a48; color:white; border-radius:4px;")
                else:
                    button.setText("⚑" if cell in self.game.flags else "")
                    button.setStyleSheet("background:#6f6687; color:white; border-radius:4px;")
