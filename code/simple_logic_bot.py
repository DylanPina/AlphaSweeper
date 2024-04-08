from utils import GameResult
from minesweeper import Minesweeper
from typing import Tuple
import random


class MinesweeperBot:
    def __init__(self, game: Minesweeper):
        self.game: Minesweeper = game
        self.width: int = game.width
        self.height: int = game.height
        self.cells_remaining: set[Tuple[int, int]] = {
            (x, y) for x in range(self.width) for y in range(self.height)
        }
        self.inferred_safe: set = set()
        self.inferred_mine: set = set()
        self.clues: dict = {}

    def select_cell(self) -> Tuple[int, int]:
        """Selects a cell to uncover, preferring safe cells, then random."""

        if self.inferred_safe:
            return self.inferred_safe.pop()

        remaining_minus_inferred = self.cells_remaining - self.inferred_mine
        random.shuffle(list(remaining_minus_inferred))
        return remaining_minus_inferred.pop()

    def update_knowledge(self, cell: Tuple[int, int]):
        """Updates the bot's knowledge based on the uncovered cell."""

        x, y = cell
        clue = self.game.adjacent_mines(x, y)
        self.clues[cell] = clue
        self.cells_remaining.discard(cell)

        if clue == 0:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = cell[0] + dx, cell[1] + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        self.inferred_safe.add((nx, ny))

    def infer_cells(self):
        """Infers and marks cells as safe or mines based on the clues."""

        for cell, clue in self.clues.items():
            x, y = cell
            neighbors = [
                (x + dx, y + dy)
                for dx in range(-1, 2)
                for dy in range(-1, 2)
                if 0 <= x + dx < self.width
                and 0 <= y + dy < self.height
                and (x + dx, y + dy) != cell
            ]

            unrevealed_neighbors = [n for n in neighbors if n in self.cells_remaining]
            inferred_mines_count = sum(1 for n in neighbors if n in self.inferred_mine)
            safe_count = sum(
                1
                for n in neighbors
                if n in self.inferred_safe or n not in self.cells_remaining
            )

            # If the number of mines equals the clue, all other neighbors are safe
            if clue - inferred_mines_count == len(unrevealed_neighbors):
                self.inferred_mine.update(unrevealed_neighbors)
            # If the total safe spots equals the total neighbors minus the clue, all unrevealed are safe
            if (8 - clue) - safe_count == len(unrevealed_neighbors):
                self.inferred_safe.update(unrevealed_neighbors)

    def play_turn(self):
        """Plays a single turn of Minesweeper."""

        selected_cell = self.select_cell()
        result = self.game.uncover(selected_cell)
        if result == GameResult.MINE:
            print(f"Hit a mine at {selected_cell}")
            return result
        self.update_knowledge(selected_cell)
        self.infer_cells()
        return True
