import logging
import random
from typing import List, Tuple
from common.utils import GameResult
from game.minesweeper import Minesweeper


class LogicBot:

    def __init__(self, game: Minesweeper, logger):
        self.game: Minesweeper = game
        self.width: int = game.width
        self.height: int = game.height
        self.inferred_safe: set = set()
        self.inferred_mines: set = set()
        self.clues: dict = {}
        self.moves = []
        self.board_states = []
        self.first_move = True
        self.logger = logger

    def select_cell(self) -> Tuple[int, int]:
        """Selects a cell to uncover, preferring safe cells, then random."""

        if not self.moves:
            return (self.height // 2, self.width // 2)

        if self.inferred_safe:
            return self.inferred_safe.pop()

        return random.choice(tuple(self.game.remaining_cells))

    def infer_cells(self):
        """Infers and marks cells as safe or mines based on the clues."""

        for cell, clue in self.game.clues.items():
            (
                neighbors,
                revealed_neighbors,
                unrevealed_neighbors,
                inferred_safe_neighbors,
                inferred_mines_neighbors,
            ) = self.get_cell_inference_data(cell)

            safe_or_revealed_neighbors = [
                n
                for n in neighbors
                if n in inferred_safe_neighbors or n in revealed_neighbors
            ]

            if clue - len(inferred_mines_neighbors) == len(unrevealed_neighbors):
                self.inferred_mines.update(unrevealed_neighbors)
                self.game.remaining_cells.difference_update(unrevealed_neighbors)

            if (len(neighbors) - clue) - len(safe_or_revealed_neighbors) == len(
                unrevealed_neighbors
            ):
                self.inferred_safe.update(unrevealed_neighbors)
                self.game.remaining_cells.difference_update(unrevealed_neighbors)

    def play_turn(self, turn_number: int):
        """Plays a single turn of Minesweeper."""

        selected_cell = self.select_cell()
        self.moves.append(selected_cell)
        self.board_states.append(self.game.user_board)

        self.logger.debug(f"Turn {turn_number} - Selected cell: {selected_cell}")

        result = self.game.play_turn(selected_cell)

        self.infer_cells()

        if not self.game.remaining_cells:
            return GameResult.WIN

        self.logger.debug(f"# Inferred mines: {len(self.inferred_mines)}")
        self.logger.debug(f"# Inferred safe: {len(self.inferred_safe)}")

        return result

    def get_cell_inference_data(self, cell: Tuple[int, int]):
        row, col = cell
        neighbors = self.get_cell_neighbors(row, col)
        revealed_neighbors = self.get_revealed_neighbors(neighbors)
        unrevealed_neighbors = self.get_unrevealed_neighbors(neighbors)
        inferred_safe_neighbors = self.get_inferred_safe_neighbors(neighbors)
        inferred_mines_neighbors = self.get_inferred_mines_neighbors(neighbors)

        return (
            neighbors,
            revealed_neighbors,
            unrevealed_neighbors,
            inferred_safe_neighbors,
            inferred_mines_neighbors,
        )

    def get_cell_neighbors(self, row: int, col: int):
        return [
            (row + dx, col + dy)
            for dx in range(-1, 2)
            for dy in range(-1, 2)
            if row + dx in range(self.height)
            and col + dy in range(self.width)
            and (row + dx, col + dy) != (row, col)
        ]

    def get_revealed_neighbors(self, neighbors: List[Tuple[int, int]]):
        return [
            n
            for n in neighbors
            if n not in self.game.remaining_cells
            and n not in self.inferred_safe
            and n not in self.inferred_mines
        ]

    def get_unrevealed_neighbors(self, neighbors: List[Tuple[int, int]]):
        return [
            n
            for n in neighbors
            if n in self.game.remaining_cells
            and n not in self.inferred_safe
            and n not in self.inferred_mines
        ]

    def get_inferred_safe_neighbors(self, neighbors: List[Tuple[int, int]]):
        return [n for n in neighbors if n in self.inferred_safe]

    def get_inferred_mines_neighbors(self, neighbors: List[Tuple[int, int]]):
        return [n for n in neighbors if n in self.inferred_mines]
