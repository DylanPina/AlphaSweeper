from random import randint
from game.minesweeper import Minesweeper
from common.utils import GameResult
from .logic_bot import LogicBot
from common.config import close_logger, setup_logger


class LogicBotRunner:

    def __init__(
        self,
        games: int,
        width: int,
        height: int,
        mines: int | None,
        task: str,
        stop_on_hit=True,
    ):
        self.games = games
        self.width = width
        self.height = height
        self.mines = mines
        self.task = task
        self.stop_on_hit = stop_on_hit
        self.results = []
        self.moves = []
        self.board_states = []
        self.label_board = []
        self.logger = setup_logger(
            name=f"{task} Logic Bot", task=task, log_file="logic_bot"
        )

    def run(self):
        """Runs the logic bot and returns the results"""

        self.logger.info(f"Running logic bot with {self.games} games")
        self.logger.info(f"Board: {self.width}x{self.height} with {self.mines} mines")

        for game_number in range(int(self.games)):
            self.logger.debug(f"Starting game #{game_number + 1}...")

            mines = self.mines if self.mines else randint(10, 50)

            game = Minesweeper(self.width, self.height, mines, self.logger)
            bot = LogicBot(game, self.logger)

            result, turn = None, 0
            while turn < (self.width * self.height):
                result = bot.play_turn(turn)
                turn += 1

                if (
                    (result == GameResult.MINE)
                    and self.stop_on_hit
                    or result == GameResult.WIN
                ):
                    break

                game.print_board(reveal=False)

            for board_state in game.board_states:
                self.board_states.append(board_state)
                self.label_board.append(game.label_board)

            self.moves.append(turn)
            self.results.append(1 if result == GameResult.WIN else 0)

        win_rate = sum(self.results) / len(self.results)
        avg_moves = sum(self.moves) / len(self.moves)

        self.logger.info(f"Win rate: {win_rate:.2f} | Average moves: {avg_moves:.2f}")
        close_logger(self.logger)

        return {
            "board_states": self.board_states,
            "label_boards": self.label_board,
            "moves": self.moves,
            "results": self.results,
            "win_rate": win_rate,
            "average_turns": avg_moves,
        }
