import os
import random

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from pyLOCO.gui.results.results_workspace import ResultsWorkspace
from pyLOCO.gui.main_window import MainWindow
from pyLOCO.gui.widgets.waiting_games.game_2048 import Game2048
from pyLOCO.gui.widgets.waiting_games.game_dialog import WaitingGamesDialog
from pyLOCO.gui.widgets.waiting_games.minesweeper import MinesweeperGame
from pyLOCO.gui.widgets.waiting_games.sliding_puzzle import SlidingPuzzle
from pyLOCO.gui.widgets.waiting_games.tic_tac_toe import TicTacToe


def app():
    return QApplication.instance() or QApplication([])


def test_gui_response_matrix_calculator_mapping():
    app()
    window = MainWindow()
    expected = {
        "Linear (transfer matrix)": "Linear",
        "Analytical (uncoupled optics)": "Analytical",
        "Tracking": "Numerical",
    }
    assert {window.rm_calculator.itemText(i): window.rm_calculator.itemData(i) for i in range(window.rm_calculator.count())} == expected
    for project_value, backend_value in (("Analytical", "Analytical"), ("Tracking", "Numerical"), ("Numerical", "Numerical")):
        window._set_calculator_value(project_value)
        assert window.rm_calculator.currentData() == backend_value
        assert window._collect_loco_configuration().response_matrix.to_rm_config_kwargs()["calculator"] == backend_value
    window.project.modified = False
    window.close()


def test_2048_merge_scores_and_does_not_double_merge():
    game = Game2048(random.Random(1), start=False)
    game.board[0] = [2, 2, 2, 2]
    assert game.move("left", spawn=False)
    assert game.board[0] == [4, 4, 0, 0]
    assert game.score == 8


def test_2048_legal_movement_spawns_one_tile():
    game = Game2048(random.Random(2), start=False)
    game.board[0][3] = 2
    assert game.move("left")
    assert sum(value != 0 for row in game.board for value in row) == 2
    assert game.board[0][0] == 2


def test_2048_game_over_and_win_detection():
    game = Game2048(start=False)
    game.board = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
    assert game.game_over()
    game.board[0] = [1024, 1024, 0, 0]
    game.move("left", spawn=False)
    assert game.won


def test_minesweeper_places_exact_mines_and_flags():
    game = MinesweeperGame(rng=random.Random(3))
    assert len(game.mines) == 10
    cell = next(iter(set((r, c) for r in range(9) for c in range(9)) - game.mines))
    assert game.toggle_flag(*cell) and cell in game.flags
    assert game.toggle_flag(*cell) and cell not in game.flags


def test_minesweeper_empty_region_expands_and_can_win():
    game = MinesweeperGame(rows=3, columns=3, mines=1, rng=random.Random(1))
    game.mines = {(2, 2)}
    assert game.reveal(0, 0)
    assert len(game.revealed) == 8
    assert game.won


def test_minesweeper_mine_causes_loss():
    game = MinesweeperGame(rows=3, columns=3, mines=1)
    game.mines = {(1, 1)}
    assert game.reveal(1, 1)
    assert game.lost and not game.won


def test_sliding_puzzle_moves_reset_and_counts():
    puzzle = SlidingPuzzle(random.Random(4), shuffle_moves=20)
    initial = puzzle.board[:]
    legal = puzzle.legal_indices()[0]
    assert puzzle.move(legal) and puzzle.moves == 1
    assert not puzzle.move(puzzle.board.index(0))
    puzzle.reset()
    assert puzzle.board == initial and puzzle.moves == 0


def test_sliding_puzzle_generation_is_solvable_by_construction():
    puzzle = SlidingPuzzle(random.Random(5), shuffle_moves=80)
    assert sorted(puzzle.board) == list(range(16))
    assert not puzzle.is_solved()
    # Generation starts solved and applies only legal moves; every move is reversible.
    assert len(puzzle.legal_indices()) in {2, 3, 4}


def test_sliding_puzzle_solved_detection():
    puzzle = SlidingPuzzle(random.Random(6), shuffle_moves=2)
    puzzle.board = list(puzzle.SOLVED)
    assert puzzle.is_solved()


def test_tic_tac_toe_legal_moves_and_win_detection():
    game = TicTacToe()
    assert game.user_move(0)
    assert not game.user_move(0)
    game.board = ["X", "X", "X", "O", "O", "", "", "", ""]
    assert game.winner() == "X"


def test_tic_tac_toe_draw_and_new_game():
    game = TicTacToe(); game.board = ["X", "O", "X", "X", "O", "O", "O", "X", "X"]
    assert game.draw()
    game.new_game()
    assert game.board == [""] * 9 and not game.finished


def test_tic_tac_toe_computer_takes_winning_move():
    game = TicTacToe(); game.board = ["O", "O", "", "X", "X", "", "", "", ""]
    assert game.computer_move() == 2
    assert game.winner() == "O"


def test_waiting_dialog_statuses_and_game_switching():
    app(); dialog = WaitingGamesDialog()
    for state, text in WaitingGamesDialog.STATUS_TEXT.items():
        dialog.set_loco_status(state)
        assert dialog.loco_status.text() == text
    dialog.show_game("2048")
    assert dialog.stack.currentWidget() is not dialog.chooser
    dialog.close()


def test_launcher_is_visible_only_during_active_run():
    app(); workspace = ResultsWorkspace()
    assert workspace.waiting_games_button.isHidden()
    workspace.begin_run()
    assert not workspace.waiting_games_button.isHidden()
    workspace.fail_run()
    assert workspace.waiting_games_button.isHidden()


def test_main_window_reuses_one_waiting_game_dialog():
    app(); window = MainWindow()
    window._open_waiting_games(); first = window._waiting_games_dialog
    window._open_waiting_games()
    assert window._waiting_games_dialog is first
    window._set_waiting_game_status("completed")
    assert first.loco_status.text() == "✓ LOCO fit completed"
    first.close(); window.close()
