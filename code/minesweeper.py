import random
from typing import Tuple
from utils import GameResult


class Minesweeper:
    def __init__(self, width, height, mines):
        self.width = width
        self.height = height
        self.total_mines = mines
        self.total_uncovered = 0
        self.mines = set()
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

        all_cells = set((x, y) for x in range(self.height) for y in range(self.width))
        mineable_cells = list(all_cells - safe_zone)
        self.mines = set(random.sample(mineable_cells, self.total_mines))
        self.initialize_mine_board()

    def initialize_mine_board(self):
        for mine in self.mines:
            x, y = mine
            self.mine_board[x][y] = -1
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = x + dx, y + dy
                    if (
                        nx in range(self.height)
                        and ny in range(self.width)
                        and self.mine_board[nx][ny] != -1
                    ):
                        self.mine_board[nx][ny] += 1

    def print_board(self, reveal=False):
        for x in range(self.height):
            for y in range(self.width):
                print(
                    (
                        self.user_board[x][y]
                        if not reveal
                        else (
                            self.mine_board[x][y]
                            if self.mine_board[x][y] != -1
                            else "M"
                        )
                    ),
                    end=" ",
                )
            print("\n")

    def uncover(self, cell: Tuple[int, int]):
        x, y = cell
        if x not in range(self.height) or y not in range(self.width):
            return GameResult.OUT_OF_BOUNDS
        if self.user_board[x][y] != "?":
            return GameResult.ALREADY_UNCOVERED
        if self.mine_board[x][y] == -1:
            self.user_board[x][y] = "M"
            return GameResult.MINE

        self.open_adjacent_cells(x, y)
        return GameResult.OK

    def open_adjacent_cells(self, x, y):
        if x not in range(self.height) or y not in range(self.width):
            return
        if self.mine_board[x][y] == -1 or self.user_board[x][y] != "?":
            return

        self.total_uncovered += 1
        self.user_board[x][y] = str(self.mine_board[x][y])
        if self.mine_board[x][y] == 0:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    self.open_adjacent_cells(x + dx, y + dy)

    def play_turn(self, cell: Tuple[int, int]) -> GameResult:
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

        if self.total_uncovered == self.width * self.height - len(self.mines):
            print("You win!")
            self.print_board(reveal=False)

        return result

    def play_interactive(self):
        while True:
            self.print_board()
            action = input("Enter 'u x y' to uncover or 'q' to quit: ")

            if action == "q":
                break
            elif not action.startswith("u "):
                print("Invalid action. Please try again.")
                continue

            _, x, y = action.split()
            result = self.play_turn((int(x), int(y)))
            if result == GameResult.MINE:
                break


if __name__ == "__main__":
    game = Minesweeper(10, 10, 20)
    game.play_interactive()
