import logging
from game.minesweeper import Minesweeper
from logic_bot import LogicBot
from common.utils import GameResult
from common.config import init_logging


class LogicBotRunner:

    def __init__(self, log_level, games, width, height, mines):
        self.log_level = log_level
        self.games = games
        self.width = width
        self.height = height
        self.mines = mines
        self.results = []
        self.moves = []
        self.board_states = []
        self.label_board = []

    def run(self):
        """Runs the logic bot and returns the results"""

        init_logging(self.log_level)

        for game_number in range(int(self.games)):
            logging.debug(f"Game {game_number + 1}")

            game = Minesweeper(int(self.width), int(self.height), int(self.mines))
            bot = LogicBot(game)
            game.print_board()

            result, turn = None, 0
            while turn < (int(self.width) * int(self.height)):
                result = bot.play_turn(turn)
                turn += 1

                if result == GameResult.MINE or result == GameResult.WIN:
                    break

                self.board_states.append(game.user_board)
                self.label_board.append(game.safe_board)

            if turn == (int(self.width) * int(self.height)):
                logging.critical("Number of turns have exceeding the number of cells.")

            self.moves.append(turn)
            self.results.append(1 if result == GameResult.WIN else 0)

        win_rate = sum(self.results) / len(self.results)
        avg_moves = sum(self.moves) / len(self.moves)

        return (
            self.board_states,
            self.label_board,
            self.moves,
            self.results,
            win_rate,
            avg_moves,
        )
