import logging
import random
from typing import Tuple
from common.utils import GameResult


class Minesweeper:
    def __init__(self, width, height, mines):
        self.width = width
        self.height = height
        self.total_mines = mines
        self.mines = set()
        self.clues = dict()
        self.remaining_cells = set()
        self.user_board = [[-2 for _ in range(width)] for _ in range(height)]
        self.revealed_board = [[0 for _ in range(width)] for _ in range(height)]
        self.safe_board = [[1 for _ in range(width)] for _ in range(height)]

    def place_mines(self, first_cell: Tuple[int, int]):
        """Place mines on the board, ensuring the first cell and its neighbors are safe."""

        logging.debug("Placing mines...")

        safe_zone = {first_cell}
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue

                nx, ny = first_cell[0] + dx, first_cell[1] + dy
                if nx in range(self.height) and ny in range(self.width):
                    safe_zone.add((nx, ny))

        self.revealed_board[first_cell[0]][first_cell[1]] = 0
        self.remaining_cells = set(
            (row, col) for row in range(self.height) for col in range(self.width)
        )

        mineable_cells = list(self.remaining_cells - safe_zone)
        self.mines = set(random.sample(mineable_cells, self.total_mines))
        self.initialize_board()

    def initialize_board(self):
        """Initialize the board with the number of mines adjacent to each cell and where the mine cells are."""

        for mine in self.mines:
            row, col = mine
            self.revealed_board[row][col] = -1
            self.safe_board[row][col] = 0
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue

                    nx, ny = row + dx, col + dy
                    if (
                        nx in range(self.height)
                        and ny in range(self.width)
                        and self.revealed_board[nx][ny] != -1
                    ):
                        self.revealed_board[nx][ny] += 1

    def print_board(self, reveal=False):
        """Prints the board to the logs"""

        board_string = ""
        for row in range(self.height):
            board_row = []
            for col in range(self.width):
                cell_display = (
                    self.user_board[row][col]
                    if not reveal
                    else (
                        self.revealed_board[row][col]
                        if self.revealed_board[row][col] != -1
                        else "M"
                    )
                )
                board_row.append(str(cell_display))
            board_string += "\n" if row == 0 else "" + " ".join(board_row) + "\n"
        logging.debug(board_string)

    def uncover(self, cell: Tuple[int, int]):
        """Uncovers a cell"""

        row, col = cell
        if row not in range(self.height) or col not in range(self.width):
            return GameResult.OUT_OF_BOUNDS
        if self.user_board[row][col] != -2:
            return GameResult.ALREADY_UNCOVERED
        if (row, col) in self.mines:
            self.user_board[row][col] = -1
            return GameResult.MINE

        self.open_adjacent_cells(row, col)

        if not self.remaining_cells:
            return GameResult.WIN

        logging.debug(f"Remaining cells: {len(self.remaining_cells)}")
        return GameResult.OK

    def open_adjacent_cells(self, row, col):
        """Opens adjacent cells for the given cell according to rules of minesweeper"""

        if (
            row not in range(self.height)
            or col not in range(self.width)
            or (row, col) in self.mines
            or (row, col) not in self.remaining_cells
        ):
            return

        self.user_board[row][col] = self.revealed_board[row][col]
        self.clues[(row, col)] = self.revealed_board[row][col]
        self.remaining_cells.remove((row, col))

        if self.revealed_board[row][col] == 0:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    self.open_adjacent_cells(row + dx, col + dy)

    def play_turn(self, cell: Tuple[int, int]) -> GameResult:
        """Plays a single turn of Minesweeper."""

        if not self.mines:
            self.place_mines(cell)

        result = self.uncover(cell)
        if result == GameResult.OUT_OF_BOUNDS:
            logging.debug("That cell is out of bounds. Please try again.")
        elif result == GameResult.ALREADY_UNCOVERED:
            logging.debug("That cell is already uncovered. Please try again.")
        elif result == GameResult.MINE:
            logging.debug("Mine hit! Game Over!")
            self.print_board(reveal=True)
        elif result == GameResult.WIN:
            logging.debug("You win!")
            self.print_board(reveal=True)

        return result

    def play_interactive(self):
        """Plays the game in interactive mode (through the console)"""

        while True:
            self.print_board()
            action = input("Enter 'u row col' to uncover or 'q' to quit: ")

            if action == "q":
                break
            elif not action.startswith("u "):
                logging.debug("Invalid action. Please try again.")
                continue

            _, row, col = action.split()
            result = self.play_turn((int(row), int(col)))
            if result == GameResult.MINE or result == GameResult.WIN:
                break
