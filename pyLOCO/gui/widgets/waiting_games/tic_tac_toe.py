"""Tic-tac-toe against a small deterministic minimax opponent."""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class TicTacToe:
    WINS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
            (2, 5, 8), (0, 4, 8), (2, 4, 6))

    def __init__(self) -> None: self.new_game()
    def new_game(self) -> None: self.board = [""] * 9; self.finished = False

    @classmethod
    def winner_for(cls, board: list[str]) -> str | None:
        for a, b, c in cls.WINS:
            if board[a] and board[a] == board[b] == board[c]: return board[a]
        return None

    def winner(self) -> str | None: return self.winner_for(self.board)
    def draw(self) -> bool: return all(self.board) and self.winner() is None

    def user_move(self, index: int) -> bool:
        if self.finished or not 0 <= index < 9 or self.board[index]: return False
        self.board[index] = "X"
        if self.winner() or self.draw(): self.finished = True
        return True

    def computer_move(self) -> int | None:
        if self.finished: return None
        choices = [index for index, value in enumerate(self.board) if not value]
        if not choices: self.finished = True; return None
        best = max(choices, key=lambda index: (self._score_move(index), -index))
        self.board[best] = "O"
        if self.winner() or self.draw(): self.finished = True
        return best

    def _score_move(self, index: int) -> int:
        board = self.board[:]; board[index] = "O"
        return self._minimax(board, False)

    @classmethod
    def _minimax(cls, board: list[str], maximizing: bool) -> int:
        winner = cls.winner_for(board)
        if winner == "O": return 10
        if winner == "X": return -10
        choices = [index for index, value in enumerate(board) if not value]
        if not choices: return 0
        scores = []
        mark = "O" if maximizing else "X"
        for index in choices:
            candidate = board[:]; candidate[index] = mark
            scores.append(cls._minimax(candidate, not maximizing))
        return max(scores) if maximizing else min(scores)


class TicTacToeWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent); self.game = TicTacToe()
        title = QLabel("Tic-Tac-Toe"); title.setObjectName("gameTitle"); self.status = QLabel("Your turn — you are X")
        grid = QGridLayout(); grid.setSpacing(6); self.cells = []
        for index in range(9):
            button = QPushButton(); button.setFixedSize(92, 92)
            button.clicked.connect(lambda checked=False, i=index: self._play(i))
            button.setStyleSheet("font-size:28px; font-weight:700; border-radius:8px;")
            grid.addWidget(button, index // 3, index % 3); self.cells.append(button)
        new_button = QPushButton("New Game"); new_button.clicked.connect(self._new)
        layout = QVBoxLayout(self); layout.addWidget(title); layout.addWidget(self.status); layout.addLayout(grid); layout.addWidget(new_button)
        self.refresh()

    def _new(self) -> None: self.game.new_game(); self.refresh()
    def _play(self, index: int) -> None:
        if not self.game.user_move(index): return
        if not self.game.finished: self.game.computer_move()
        self.refresh()

    def refresh(self) -> None:
        winner = self.game.winner()
        if winner == "X": message = "You win!"
        elif winner == "O": message = "Computer wins!"
        elif self.game.draw(): message = "Draw"
        else: message = "Your turn — you are X"
        self.status.setText(message)
        for index, value in enumerate(self.game.board):
            self.cells[index].setText(value); self.cells[index].setEnabled(not self.game.finished and not value)
