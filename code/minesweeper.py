import random
from typing import Tuple
from utils import GameResult


class Minesweeper:
    def __init__(self, width, height, mines):
        self.width = width
        self.height = height
        self.total_mines = mines
        self.mines = set()
        self.remaining_cells = set()
        self.user_board = [["?" for _ in range(width)] for _ in range(height)]
        self.mine_board = [[0 for _ in range(width)] for _ in range(height)]

    def place_mines(self, first_cell: Tuple[int, int]):
        """Place mines on the board, ensuring the first cell and its neighbors are safe."""

        safe_zone = {first_cell}
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = first_cell[0] + dx, first_cell[1] + dy
                if nx in range(self.height) and ny in range(self.width):
                    safe_zone.add((nx, ny))
        self.mine_board[first_cell[0]][first_cell[1]] = 0

        all_cells = set(
            (row, col) for row in range(self.height) for col in range(self.width)
        )
        mineable_cells = list(all_cells - safe_zone)
        self.mines = set(random.sample(mineable_cells, self.total_mines))
        self.remaining_cells = all_cells - self.mines
        self.initialize_mine_board()

    def initialize_mine_board(self):
        for mine in self.mines:
            row, col = mine
            self.mine_board[row][col] = -1
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = row + dx, col + dy
                    if (
                        nx in range(self.height)
                        and ny in range(self.width)
                        and self.mine_board[nx][ny] != -1
                    ):
                        self.mine_board[nx][ny] += 1

    def print_board(self, reveal=False):
        for row in range(self.height):
            for col in range(self.width):
                print(
                    (
                        self.user_board[row][col]
                        if not reveal
                        else (
                            self.mine_board[row][col]
                            if self.mine_board[row][col] != -1
                            else "M"
                        )
                    ),
                    end=" ",
                )
            print()

    def uncover(self, cell: Tuple[int, int]):
        row, col = cell
        if row not in range(self.height) or col not in range(self.width):
            return GameResult.OUT_OF_BOUNDS
        if self.user_board[row][col] != "?":
            return GameResult.ALREADY_UNCOVERED
        if self.mine_board[row][col] == -1:
            self.user_board[row][col] = "M"
            return GameResult.MINE

        self.open_adjacent_cells(row, col)

        if not self.remaining_cells:
            return GameResult.WIN

        print(f"Remaining cells: {len(self.remaining_cells)}")
        return GameResult.OK

    def open_adjacent_cells(self, row, col):
        if row not in range(self.height) or col not in range(self.width):
            return
        if self.mine_board[row][col] == -1 or self.user_board[row][col] != "?":
            return

        self.user_board[row][col] = str(self.mine_board[row][col])
        self.remaining_cells.remove((row, col))

        if self.mine_board[row][col] == 0:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    self.open_adjacent_cells(row + dx, col + dy)

    def play_turn(self, cell: Tuple[int, int]) -> GameResult:
        """Plays a single turn of Minesweeper."""

        if len(self.mines) == 0:
            self.place_mines(cell)

        result = self.uncover(cell)
        if result == GameResult.OUT_OF_BOUNDS:
            print("That cell is out of bounds. Please try again.")
        elif result == GameResult.ALREADY_UNCOVERED:
            print("That cell is already uncovered. Please try again.")
        elif result == GameResult.MINE:
            print("Game Over!")
            self.print_board(reveal=True)
        elif result == GameResult.WIN:
            print("You win!")
            self.print_board(reveal=False)

        return result

    def play_interactive(self):
        while True:
            self.print_board()
            action = input("Enter 'u row col' to uncover or 'q' to quit: ")

            if action == "q":
                break
            elif not action.startswith("u "):
                print("Invalid action. Please try again.")
                continue

            _, row, col = action.split()
            result = self.play_turn((int(row), int(col)))
            if result == GameResult.MINE or result == GameResult.WIN:
                break


if __name__ == "__main__":
    game = Minesweeper(10, 10, 20)
    game.play_interactive()
