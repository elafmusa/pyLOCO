"""Small 2048 model and Qt view with no timers or external assets."""

from __future__ import annotations

import random

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class Game2048:
    def __init__(self, rng: random.Random | None = None, *, start: bool = True) -> None:
        self.rng = rng or random.Random()
        self.board = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.won = False
        if start:
            self.add_tile(); self.add_tile()

    def new_game(self) -> None:
        self.board = [[0] * 4 for _ in range(4)]
        self.score = 0; self.won = False
        self.add_tile(); self.add_tile()

    def add_tile(self) -> bool:
        empty = [(r, c) for r in range(4) for c in range(4) if self.board[r][c] == 0]
        if not empty:
            return False
        row, column = self.rng.choice(empty)
        self.board[row][column] = 4 if self.rng.random() < 0.1 else 2
        return True

    @staticmethod
    def _collapse(line: list[int]) -> tuple[list[int], int]:
        values = [value for value in line if value]
        result: list[int] = []
        gained = 0
        index = 0
        while index < len(values):
            if index + 1 < len(values) and values[index] == values[index + 1]:
                merged = values[index] * 2
                result.append(merged); gained += merged; index += 2
            else:
                result.append(values[index]); index += 1
        return result + [0] * (4 - len(result)), gained

    def move(self, direction: str, *, spawn: bool = True) -> bool:
        direction = direction.lower()
        if direction not in {"left", "right", "up", "down"}:
            raise ValueError(f"Unknown 2048 direction: {direction}")
        before = [row[:] for row in self.board]
        gained = 0
        lines = self.board if direction in {"left", "right"} else [
            [self.board[row][column] for row in range(4)] for column in range(4)
        ]
        transformed = []
        for line in lines:
            oriented = list(reversed(line)) if direction in {"right", "down"} else line
            collapsed, points = self._collapse(oriented)
            transformed.append(list(reversed(collapsed)) if direction in {"right", "down"} else collapsed)
            gained += points
        if direction in {"left", "right"}:
            self.board = transformed
        else:
            self.board = [[transformed[column][row] for column in range(4)] for row in range(4)]
        changed = self.board != before
        if changed:
            self.score += gained
            self.won = self.won or any(2048 in row for row in self.board)
            if spawn:
                self.add_tile()
        return changed

    def game_over(self) -> bool:
        if any(0 in row for row in self.board):
            return False
        return not any(
            self.board[row][column] == self.board[nr][nc]
            for row in range(4) for column in range(4)
            for nr, nc in ((row + 1, column), (row, column + 1))
            if nr < 4 and nc < 4
        )


class Game2048Widget(QWidget):
    COLORS = {0: "#34374e", 2: "#eee4da", 4: "#ede0c8", 8: "#d6b07a", 16: "#c48b62",
              32: "#b76a55", 64: "#a94c46", 128: "#8e70b8", 256: "#7652a7",
              512: "#65449a", 1024: "#563786", 2048: "#7e57c2"}

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.game = Game2048()
        self.setFocusPolicy(Qt.StrongFocus)
        title = QLabel("2048"); title.setObjectName("gameTitle")
        self.status = QLabel()
        grid = QGridLayout(); grid.setSpacing(6)
        self.tiles = []
        for row in range(4):
            row_tiles = []
            for column in range(4):
                tile = QLabel(); tile.setAlignment(Qt.AlignCenter); tile.setMinimumSize(68, 68)
                grid.addWidget(tile, row, column); row_tiles.append(tile)
            self.tiles.append(row_tiles)
        button = QPushButton("New Game"); button.clicked.connect(self._new)
        layout = QVBoxLayout(self); layout.addWidget(title); layout.addWidget(self.status); layout.addLayout(grid); layout.addWidget(button)
        self.refresh()

    def _new(self) -> None:
        self.game.new_game(); self.refresh(); self.setFocus()

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        directions = {Qt.Key_Left: "left", Qt.Key_A: "left", Qt.Key_Right: "right", Qt.Key_D: "right",
                      Qt.Key_Up: "up", Qt.Key_W: "up", Qt.Key_Down: "down", Qt.Key_S: "down"}
        direction = directions.get(event.key())
        if direction:
            self.game.move(direction); self.refresh(); event.accept(); return
        super().keyPressEvent(event)

    def refresh(self) -> None:
        message = f"Score: {self.game.score}"
        if self.game.won: message += "   🎉 2048 reached!"
        elif self.game.game_over(): message += "   No moves remaining"
        self.status.setText(message)
        for row in range(4):
            for column in range(4):
                value = self.game.board[row][column]
                color = self.COLORS.get(value, "#4b2878")
                text = "" if value == 0 else str(value)
                foreground = "#252638" if value in {2, 4, 8} else "#ffffff"
                tile = self.tiles[row][column]; tile.setText(text)
                tile.setStyleSheet(f"background:{color}; color:{foreground}; border-radius:8px; font-size:18px; font-weight:700;")
