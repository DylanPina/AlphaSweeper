from network.network import Network
from common.config import close_logger, setup_logger
from game.minesweeper import Minesweeper
from .network_bot import NetworkBot
from common.utils import GameResult
from random import randint


class NetworkBotRunner:

    def __init__(
        self,
        network: Network,
        games: int,
        width: int,
        height: int,
        mines: int | None,
        task: str,
    ):
        self.network = network
        self.games = games
        self.width = width
        self.height = height
        self.mines = mines
        self.task = task
        self.results = []
        self.moves = []
        self.board_states = []
        self.label_board = []
        self.logger = setup_logger(
            f"{task} Network Bot", task=task, log_file="network_bot"
        )

    def run(self):
        """Runs the network bot and returns the results"""

        self.logger.info(f"Running network bot with {self.games} games")

        for game_number in range(int(self.games)):
            self.logger.debug(f"Starting game #{game_number + 1}...")

            mines = self.mines if self.mines else randint(0, 270)

            game = Minesweeper(self.width, self.height, mines, self.logger)
            bot = NetworkBot(self.network, game, self.logger)

            result, turn = None, 0
            while turn < (self.width * self.height):
                result = bot.play_turn(turn)
                turn += 1

                if len(game.remaining_cells) == len(game.mines):
                    self.logger.debug("Network bot has won!")
                    result = GameResult.WIN
                    break

                if result == GameResult.MINE:
                    break

                game.print_board(reveal=False)

                self.board_states.append(game.user_board)
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
