from common.config import close_logger, setup_logger, base_dir
from game.minesweeper import Minesweeper
from .network_bot import NetworkBot
from common.utils import GameResult


class NetworkBotRunner:

    def __init__(self, network, games, width, height, mines):
        self.network = network
        self.games = games
        self.width = width
        self.height = height
        self.mines = mines
        self.results = []
        self.moves = []
        self.board_states = []
        self.label_board = []
        self.logger = setup_logger(
            "Network Bot", f"{base_dir}/logs/task_1/network_bot.log"
        )

    def run(self):
        """Runs the network bot and returns the results"""

        for game_number in range(int(self.games)):
            self.logger.info(f"Starting game #{game_number + 1}...")

            game = Minesweeper(
                int(self.width), int(self.height), int(self.mines), self.logger
            )
            bot = NetworkBot(self.network, game, self.logger)

            result, turn = None, 0
            while turn < (int(self.width) * int(self.height)):
                result = bot.play_turn(turn)
                turn += 1

                if len(game.remaining_cells) == len(game.mines):
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

        return (
            self.board_states,
            self.label_board,
            self.moves,
            self.results,
            win_rate,
            avg_moves,
        )
