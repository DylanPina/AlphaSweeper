from common.config import close_logger, setup_logger
from game.minesweeper import Minesweeper


if __name__ == "__main__":
    logger = setup_logger("minesweeper", "interactive.log")
    game = Minesweeper(9, 9, 10, logger)
    game.play_interactive()
    close_logger(logger)
