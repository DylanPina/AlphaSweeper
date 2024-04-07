from minesweeper import Minesweeper
from typing import Tuple

class MinesweeperBot:
    def __init__(self, game: Minesweeper):
        self.game = game
        self.width = game.width
        self.height = game.height
        self.cells_remaining = {(x, y) for x in range(self.width) for y in range(self.height)}
        self.inferred_safe = set()
        self.inferred_mine = set()
        self.clue_numbers = {}

    def select_cell(self) -> Tuple[int, int]:
        """Selects a cell to uncover, preferring safe cells, then random."""

        if self.inferred_safe:
            return self.inferred_safe.pop()
        # Random selection from remaining cells, excluding inferred mines.
        return (self.cells_remaining - self.inferred_mine).pop()

    def update_knowledge(self, cell: Tuple[int, int], clue: int):
        """Updates the bot's knowledge based on the uncovered cell."""

        self.clue_numbers[cell] = clue
        self.cells_remaining.discard(cell)
        # Immediately infer surrounding cells if clue is 0

        if clue == 0:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = cell[0] + dx, cell[1] + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        self.inferred_safe.add((nx, ny))

    def infer_cells(self):
        """Infers and marks cells as safe or mines based on the clues."""

        for cell, clue in self.clue_numbers.items():
            x, y = cell
            neighbors = [(x + dx, y + dy) for dx in range(-1, 2) for dy in range(-1, 2)
                         if 0 <= x + dx < self.width and 0 <= y + dy < self.height and (x + dx, y + dy) != cell]

            # Unrevealed neighbors
            unrevealed_neighbors = [n for n in neighbors if n in self.cells_remaining]
            # Count already inferred mines around the current cell
            inferred_mines_count = sum(1 for n in neighbors if n in self.inferred_mine)
            # Count safe spots around the current cell
            safe_count = sum(1 for n in neighbors if n in self.inferred_safe or n not in self.cells_remaining)

            # If the number of mines equals the clue, all other neighbors are safe
            if clue - inferred_mines_count == len(unrevealed_neighbors):
                self.inferred_mine.update(unrevealed_neighbors)
            # If the total safe spots equals the total neighbors minus the clue, all unrevealed are safe
            if (8 - clue) - safe_count == len(unrevealed_neighbors):
                self.inferred_safe.update(unrevealed_neighbors)

    def play_turn(self):
        """Plays a single turn of Minesweeper."""

        cell = self.select_cell()
        result, clue = self.game.open_cell(cell)
        if result == "mine":
            print("Hit a mine at:", cell)
            return False
        self.update_knowledge(cell, clue)
        self.infer_cells()
        return True