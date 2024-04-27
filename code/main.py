import argparse
import logging
from bots.network_bot.network_bot_runner import NetworkBotRunner
from common.config import init_logging
from tasks.task_1.task_1 import Task1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-log")
    parser.add_argument("-train_data_file")
    parser.add_argument("-test_data_file")
    parser.add_argument("-test_games")
    parser.add_argument("-train_games")
    parser.add_argument("-width")
    parser.add_argument("-height")
    parser.add_argument("-mines")

    args = parser.parse_args()
    (
        log_level,
        test_data_file,
        train_data_file,
        train_games,
        test_games,
        width,
        height,
        mines,
    ) = (
        args.log,
        args.train_data_file,
        args.test_data_file,
        args.train_games,
        args.test_games,
        args.width,
        args.height,
        args.mines,
    )

    init_logging(log_level)

    task = Task1(
        log_level,
        test_data_file,
        train_data_file,
        test_games,
        train_games,
        width,
        height,
        mines,
    )

    train_losses, train_accuracies, test_losses, test_accuracies, elapsed_times = (
        task.train()
    )
    task.plot(train_losses, test_losses, train_accuracies, test_accuracies)

    runner = NetworkBotRunner(
        log_level=log_level,
        network=task.network,
        games=test_games,
        width=width,
        height=height,
        mines=mines,
    )
    runner.run()
