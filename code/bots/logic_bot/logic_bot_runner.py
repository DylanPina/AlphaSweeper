from game.minesweeper import Minesweeper
from common.utils import GameResult
from .logic_bot import LogicBot
from common.config import close_logger, setup_logger, base_dir


class LogicBotRunner:

    def __init__(self, games, width, height, mines, stop_on_hit=True):
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

        self.logger.info(f"Running logic bot with {self.games} games")
        self.logger.info(f"Board: {self.width}x{self.height} with {self.mines} mines")

        for game_number in range(int(self.games)):
            self.logger.debug(f"Starting game #{game_number + 1}...")

            game = Minesweeper(
                int(self.width), int(self.height), int(self.mines), self.logger
            )
            bot = LogicBot(game, self.logger)

            result, turn = None, 0
            while turn < (int(self.width) * int(self.height)):
                result = bot.play_turn(turn)
                turn += 1

                if (
                    (result == GameResult.MINE)
                    and self.stop_on_hit
                    or result == GameResult.WIN
                ):
                    break

                game.print_board(reveal=False)

            print(f"Game {game_number + 1} result: {result}")
            for i, board_state in enumerate(game.board_states):
                print(f"Board state {i + 1}:")
                for row in board_state:
                    print(row)
                self.board_states.append(board_state)
                self.label_board.append(game.label_board)

            self.moves.append(turn)
            self.results.append(1 if result == GameResult.WIN else 0)

        win_rate = sum(self.results) / len(self.results)
        avg_moves = sum(self.moves) / len(self.moves)

        self.logger.info(
            f"Logic bot finished playing. Win Rate: {win_rate} | Avg. Moves: {avg_moves}"
        )
        close_logger(self.logger)

        return (
            self.board_states,
            self.label_board,
            self.moves,
            self.results,
            win_rate,
            avg_moves,
        )
