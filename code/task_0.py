import argparse
from common.config import configure_logging
from bots.logic_bot.logic_bot_runner import LogicBotRunner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-log")
    parser.add_argument("-games")
    parser.add_argument("-width")
    parser.add_argument("-height")
    parser.add_argument("-mines")

    args = parser.parse_args()
    log_level = args.log
    games = int(args.games)
    width = int(args.width)
    height = int(args.height)
    mines = int(args.mines)

    configure_logging(log_level)
    runner = LogicBotRunner(games, width, height, mines)
    runner.run()
