import logging
from minesweeper import Minesweeper
from logic_bot import LogicBot
from utils import GameResult
from config import init_logging, difficulty_levels


class LogicBotRunner:

    def __init__(self, log_level, difficulty, games):
        self.log_level = log_level
        self.difficulty = difficulty
        self.games = games
        self.width, self.height, self.mines = difficulty_levels[difficulty]
        self.results = {}
        self.moves = {}

    def run(self):
        init_logging(self.log_level)
        for game_number in range(int(self.games)):
            logging.debug(f"Game {game_number + 1}")

            game = Minesweeper(int(self.height), int(self.width), int(self.mines))
            bot = LogicBot(game)
            game.print_board()

            result, turn = None, 0
            while turn < int(self.width * self.height):
                result = bot.play_turn(turn)
                turn += 1
                if result == GameResult.MINE or result == GameResult.WIN:
                    break

            if turn == int(self.width * self.height):
                logging.critical("Number of turns have exceeding the number of cells.")

            self.moves[game_number + 1] = turn
            self.results[game_number + 1] = 1 if result == GameResult.WIN else 0

        win_rate = sum(self.results.values()) / len(self.results)
        avg_moves = sum(self.moves.values()) / len(self.moves)
        return self.moves, self.results, win_rate, avg_moves
