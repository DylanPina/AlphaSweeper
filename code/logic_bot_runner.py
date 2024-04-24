import argparse
from minesweeper import Minesweeper
from logic_bot import MinesweeperBot
from utils import GameResult
from config import init_logging


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-log")
    parser.add_argument("-games")
    parser.add_argument("-size")
    parser.add_argument("-mines")

    args = parser.parse_args()
    log_level, games, size, mines = args.log, args.games, args.size, args.mines
    init_logging(log_level)

    print(f"Playing {games} games of {size}x{size} with {mines} mines.")

    for i in range(int(games)):
        print(f"Game {i + 1}")
        game = Minesweeper(10, 10, 10)
        bot = MinesweeperBot(game)
        game.print_board()
        for i in range(50):
            result = bot.play_turn(i + 1)
            if result == GameResult.MINE or result == GameResult.WIN:
                print(f"Moves: {bot.moves}")
                break
            game.print_board()
