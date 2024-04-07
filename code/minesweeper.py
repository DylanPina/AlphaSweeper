import random
from utils import UncoverResult


class Minesweeper:
    def __init__(self, width, height, mines):
        self.width = width
        self.height = height
        self.mines = set(random.sample(range(width * height), mines))
        self.covered = [[True for _ in range(width)] for _ in range(height)]

    def print_board(self, reveal=False):
        for y in range(self.height):
            for x in range(self.width):
                if reveal or not self.covered[y][x]:
                    print(
                        (
                            "M"
                            if (y * self.width + x) in self.mines
                            else self.adjacent_mines(x, y)
                        ),
                        end=" ",
                    )
                else:
                    print("?", end=" ")
            print()

    def adjacent_mines(self, x, y):
        count = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (ny * self.width + nx) in self.mines:
                        count += 1
        return count

    def uncover(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return UncoverResult.OUT_OF_BOUNDS
        if not self.covered[y][x]:
            return UncoverResult.ALREADY_UNCOVERED
        if (y * self.width + x) in self.mines:
            return UncoverResult.MINE
        self.covered[y][x] = False
        if self.adjacent_mines(x, y) == 0:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = x + dx, y + dy
                    if (
                        0 <= nx < self.width
                        and 0 <= ny < self.height
                        and self.covered[ny][nx]
                    ):
                        self.uncover(nx, ny)
        return UncoverResult.OK

    def play(self):
        while True:
            self.print_board()
            action = input("Enter 'u x y' to uncover or 'q' to quit: ")

            if action == "q":
                break
            elif not action.startswith("u "):
                print("Invalid action. Please try again.")
                continue

            _, x, y = action.split()
            x, y = int(x), int(y)
            result = self.uncover(x, y)
            if result == UncoverResult.OUT_OF_BOUNDS:
                print("That cell is out of bounds. Please try again.")
                continue
            elif result == UncoverResult.ALREADY_UNCOVERED:
                print("That cell is already uncovered. Please try again.")
                continue
            elif result == UncoverResult.MINE:
                print("Game Over!")
                self.print_board(reveal=True)
                break
            if all(
                self.covered[y][x] == (y * self.width + x in self.mines)
                for y in range(self.height)
                for x in range(self.width)
            ):
                print("Congratulations, you win!")
                break


if __name__ == "__main__":
    game = Minesweeper(10, 10, 20)
    game.play()
