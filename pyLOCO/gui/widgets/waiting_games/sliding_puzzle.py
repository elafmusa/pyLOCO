"""Guaranteed-solvable 15-puzzle model and Qt view."""

from __future__ import annotations

import random

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget


class SlidingPuzzle:
    SOLVED = tuple(range(1, 16)) + (0,)

    def __init__(self, rng: random.Random | None = None, shuffle_moves: int = 160) -> None:
        self.rng = rng or random.Random(); self.shuffle_moves = shuffle_moves
        self.board = list(self.SOLVED); self.start_board = list(self.SOLVED); self.moves = 0
        self.new_puzzle()

    def legal_indices(self) -> list[int]:
        empty = self.board.index(0); row, column = divmod(empty, 4); result = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, column + dc
            if 0 <= nr < 4 and 0 <= nc < 4: result.append(4 * nr + nc)
        return result

    def move(self, index: int, *, count: bool = True) -> bool:
        if index not in self.legal_indices(): return False
        empty = self.board.index(0); self.board[empty], self.board[index] = self.board[index], self.board[empty]
        if count: self.moves += 1
        return True

    def new_puzzle(self) -> None:
        self.board = list(self.SOLVED); previous = None
        for _ in range(self.shuffle_moves):
            choices = [index for index in self.legal_indices() if index != previous]
            empty = self.board.index(0); choice = self.rng.choice(choices)
            self.move(choice, count=False); previous = empty
        if self.is_solved():
            self.move(self.legal_indices()[0], count=False)
        self.start_board = self.board[:]; self.moves = 0

    def reset(self) -> None: self.board = self.start_board[:]; self.moves = 0
    def is_solved(self) -> bool: return tuple(self.board) == self.SOLVED


class SlidingPuzzleWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent); self.game = SlidingPuzzle()
        title = QLabel("15 Puzzle"); title.setObjectName("gameTitle"); self.status = QLabel()
        grid = QGridLayout(); grid.setSpacing(5); self.tiles = []
        for index in range(16):
            button = QPushButton(); button.setFixedSize(72, 72)
            button.clicked.connect(lambda checked=False, i=index: self._move(i))
            grid.addWidget(button, index // 4, index % 4); self.tiles.append(button)
        new_button = QPushButton("New Puzzle"); new_button.clicked.connect(self._new)
        reset_button = QPushButton("Reset"); reset_button.clicked.connect(self._reset)
        controls = QHBoxLayout(); controls.addWidget(new_button); controls.addWidget(reset_button)
        layout = QVBoxLayout(self); layout.addWidget(title); layout.addWidget(self.status); layout.addLayout(grid); layout.addLayout(controls)
        self.refresh()

    def _move(self, index: int) -> None: self.game.move(index); self.refresh()
    def _new(self) -> None: self.game.new_puzzle(); self.refresh()
    def _reset(self) -> None: self.game.reset(); self.refresh()
    def refresh(self) -> None:
        self.status.setText(f"🎉 Solved in {self.game.moves} moves!" if self.game.is_solved() else f"Moves: {self.game.moves}")
        for index, value in enumerate(self.game.board):
            button = self.tiles[index]; button.setText(str(value) if value else ""); button.setEnabled(bool(value))
            button.setStyleSheet(
                "background:#7e57c2; color:white; border-radius:7px; font-size:17px; font-weight:700;"
                if value else "background:transparent; border:1px dashed #777; border-radius:7px;"
            )
