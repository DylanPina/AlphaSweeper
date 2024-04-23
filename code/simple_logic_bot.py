from utils import GameResult
from minesweeper import Minesweeper
from typing import Tuple
import random


class MinesweeperBot:
    def __init__(self, game: Minesweeper):
        self.game: Minesweeper = game
        self.width: int = game.width
        self.height: int = game.height
        self.inferred_safe: set = set()
        self.inferred_mines: set = set()
        self.clues: dict = {}
        self.first_move = True

    def select_cell(self) -> Tuple[int, int]:
        """Selects a cell to uncover, preferring safe cells, then random."""

        if self.first_move:
            self.first_move = False
            return random.choice(
                [(x, y) for x in range(self.width) for y in range(self.height)]
            )

        if self.inferred_safe:
            print("Selecting safe cell")
            return self.inferred_safe.pop()

        if len(self.game.remaining_cells):
            print("Selecting random remaining cell")
            print(f"game.remaining_cells: {self.game.remaining_cells}")
            print(f"self.inferred_mines: {self.inferred_mines}")
            print(f"mine cells: {self.game.mines}")
            remaining_non_inferred_mine_cells = (
                self.game.remaining_cells - self.inferred_mines
            )
            if remaining_non_inferred_mine_cells:
                return random.choice(tuple(remaining_non_inferred_mine_cells))
            else:
                return random.choice(tuple(self.game.remaining_cells))
        else:
            raise ValueError("No cells remaining to select.")

    def update_knowledge(self, cell: Tuple[int, int]):
        """Updates the bot's knowledge based on the uncovered cell."""

        row, col = cell
        clue = self.game.user_board[row][col]
        self.clues[cell] = int(clue)

    def infer_cells(self):
        """Infers and marks cells as safe or mines based on the clues."""

        for cell, clue in self.clues.items():
            row, col = cell
            neighbors = [
                (row + dx, col + dy)
                for dx in range(-1, 2)
                for dy in range(-1, 2)
                if row + dx in range(self.height)
                and col + dy in range(self.width)
                and (row + dx, col + dy) != cell
            ]

            unrevealed_neighbors = [
                n for n in neighbors if n in self.game.remaining_cells
            ]
            inferred_mines_count = sum(1 for n in neighbors if n in self.inferred_mines)
            safe_count = sum(
                1
                for n in neighbors
                if n in self.inferred_safe or n not in self.game.remaining_cells
            )

            # If the number of mines equals the clue, all other neighbors are safe
            if clue - inferred_mines_count == len(unrevealed_neighbors):
                self.inferred_mines.update(unrevealed_neighbors)
            # If the total safe spots equals the total neighbors minus the clue, all unrevealed are safe
            if (8 - clue) - safe_count == len(unrevealed_neighbors):
                self.inferred_safe.update(unrevealed_neighbors)

    def play_turn(self):
        """Plays a single turn of Minesweeper."""

        selected_cell = self.select_cell()
        print(f"Selected cell: {selected_cell}")
        result = self.game.play_turn(selected_cell)
        self.update_knowledge(selected_cell)
        self.infer_cells()
        return result


if __name__ == "__main__":
    game = Minesweeper(10, 10, 10)
    bot = MinesweeperBot(game)
    game.print_board()
    for _ in range(50):
        result = bot.play_turn()
        if result == GameResult.MINE or result == GameResult.WIN:
            break
        game.print_board()
