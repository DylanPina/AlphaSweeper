from game.minesweeper import Minesweeper
from common.utils import GameResult
from .logic_bot import LogicBot
from common.config import close_logger, setup_logger, base_dir


class LogicBotRunner:

    def __init__(self, log_level, games, width, height, mines, stop_on_hit=False):
        self.log_level = log_level
        self.games = games
        self.width = width
        self.height = height
        self.mines = mines
        self.stop_on_hit = stop_on_hit
        self.results = []
        self.moves = []
        self.board_states = []
        self.label_board = []
        self.logger = setup_logger("Logic Bot", f"{base_dir}/logs/task_1/logic_bot.log")

    def run(self):
        """Runs the logic bot and returns the results"""

        for game_number in range(int(self.games)):
            self.logger.debug(f"Game {game_number + 1}")

            game = Minesweeper(
                int(self.width), int(self.height), int(self.mines), self.logger
            )
            bot = LogicBot(game, self.logger)
            game.print_board()

            result, turn = None, 0
            while turn < (int(self.width) * int(self.height)):
                result = bot.play_turn(turn)
                turn += 1

                if (
                    result == (GameResult.MINE and self.stop_on_hit)
                    or result == GameResult.WIN
                ):
                    break

                self.board_states.append(game.user_board)
                self.label_board.append(game.safe_board)

            if turn == (int(self.width) * int(self.height)):
                self.logger.critical(
                    "Number of turns have exceeding the number of cells."
                )

            close_logger(self.logger)

            self.moves.append(turn)
            self.results.append(1 if result == GameResult.WIN else 0)

        win_rate = sum(self.results) / len(self.results)
        avg_moves = sum(self.moves) / len(self.moves)
        self.logger.info(f"Win Rate: {win_rate} | Avg. Moves: {avg_moves}")

        return (
            self.board_states,
            self.label_board,
            self.moves,
            self.results,
            win_rate,
            avg_moves,
        )
