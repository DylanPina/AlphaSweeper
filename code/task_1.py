import argparse
import logging
from common.config import base_dir, configure_logging, setup_logger
from network.network import Network
from common.data_loader import MineSweeperDataLoader
from .task import Task


class Task1(Task):

    def __init__(
        self,
    ):
        super().__init__(
            setup_logger("Task 1", task="task_1", log_file="task_1"), task="task_1"
        )
        self.logger = logging.getLogger("Task 1")
        self.data_loader = MineSweeperDataLoader(logger=self.logger, task="task_1")

    def generate_data(self, train_games=50000, test_games=10000):
        """Generates the training data for easy, medium and hard games"""

        self.logger.info("Generating easy training data...")
        easy_train_data = task.run_logic_bot(
            games=train_games, width=9, height=9, mines=10
        )
        self.logger.info("Generating easy testing data...")
        easy_test_data = task.run_logic_bot(
            games=test_games, width=9, height=9, mines=10
        )

        self.logger.info("Generating medium training data...")
        medium_train_data = task.run_logic_bot(
            games=train_games, width=16, height=16, mines=40
        )
        self.logger.info("Generating medium testing data...")
        medium_test_data = task.run_logic_bot(
            games=test_games, width=16, height=16, mines=40
        )

        self.logger.info("Generating hard training data...")
        hard_train_data = task.run_logic_bot(
            games=train_games, width=30, height=16, mines=99
        )
        self.logger.info("Generating hard testing data...")
        hard_test_data = task.run_logic_bot(
            games=test_games, width=30, height=16, mines=99
        )

        self.data_loader.save(easy_train_data, "easy/train")
        self.data_loader.save(easy_test_data, "easy/test")

        self.data_loader.save(medium_train_data, "medium/train")
        self.data_loader.save(medium_test_data, "medium/test")

        self.data_loader.save(hard_train_data, "hard/train")
        self.data_loader.save(hard_test_data, "hard/test")

        return {
            "easy_train_data": easy_train_data,
            "easy_test_data": easy_test_data,
            "medium_train_data": medium_train_data,
            "medium_test_data": medium_test_data,
            "hard_train_data": hard_train_data,
            "hard_test_data": hard_test_data,
        }

    def load_data(self):
        """Loads the training data for easy, medium and hard games"""

        self.logger.debug("Loading easy data...")
        easy_train_data = self.data_loader.load("easy/train")
        easy_test_data = self.data_loader.load("easy/test")

        self.logger.debug("Loading medium data...")
        medium_train_data = self.data_loader.load("medium/train")
        medium_test_data = self.data_loader.load("medium/test")

        self.logger.debug("Loading hard data...")
        hard_train_data = self.data_loader.load("hard/train")
        hard_test_data = self.data_loader.load("hard/test")

        return {
            "easy_train_data": easy_train_data,
            "easy_test_data": easy_test_data,
            "medium_train_data": medium_train_data,
            "medium_test_data": medium_test_data,
            "hard_train_data": hard_train_data,
            "hard_test_data": hard_test_data,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-log")
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-get_data", action="store_true")
    parser.add_argument("-network_bot_games")

    args = parser.parse_args()
    (
        log_level,
        train,
        get_data,
        network_bot_games,
    ) = (args.log, args.train, args.get_data, args.network_bot_games)

    configure_logging(log_level)

    task = Task1()

    if train:
        if get_data:
            task.generate_data()

        data = task.load_data()

        easy_network = Network()
        easy_train_data, easy_test_data = (
            data["easy_train_data"],
            data["easy_test_data"],
        )
        train_losses, train_accuracies, test_losses, test_accuracies, elapsed_times = (
            task.train(
                network=easy_network,
                train_data=easy_train_data,
                test_data=easy_test_data,
                alpha=0.001,
                epochs=10,
                weight_decay=0.00001,
            )
        )
        task.save_model(easy_network, f"{base_dir}/models/task_1/easy/model.pt")
        task.plot(
            train_losses,
            test_losses,
            train_accuracies,
            test_accuracies,
            directory="easy",
        )

        medium_network = Network()
        medium_train_data, medium_test_data = (
            data["medium_train_data"],
            data["medium_test_data"],
        )
        train_losses, train_accuracies, test_losses, test_accuracies, elapsed_times = (
            task.train(
                network=medium_network,
                train_data=medium_train_data,
                test_data=medium_test_data,
                alpha=0.001,
                epochs=10,
                weight_decay=0.00001,
            )
        )
        task.save_model(easy_network, f"{base_dir}/models/task_1/medium/model.pt")
        task.plot(
            train_losses,
            test_losses,
            train_accuracies,
            test_accuracies,
            directory="medium",
        )

        hard_network = Network()
        hard_train_data, hard_test_data = (
            data["hard_train_data"],
            data["hard_test_data"],
        )
        train_losses, train_accuracies, test_losses, test_accuracies, elapsed_times = (
            task.train(
                network=hard_network,
                train_data=hard_train_data,
                test_data=hard_test_data,
                alpha=0.001,
                epochs=10,
                weight_decay=0.00001,
            )
        )
        task.save_model(easy_network, f"{base_dir}/models/task_1/hard/model.pt")
        task.plot(
            train_losses,
            test_losses,
            train_accuracies,
            test_accuracies,
            directory="hard",
        )

    print("Loading easy network...")
    easy_network = Network()
    task.load_model(easy_network, f"{base_dir}/models/task_1/easy/model.pt")
    print("Running easy network bot...")
    results = task.run_network_bot(
        easy_network, network_bot_games, width=9, height=9, mines=10
    )
    print("Finished running easy network bot!")
    print(f"Win Rate: {results['win_rate']}")
    print(f"Average Turns: {results['average_turns']}\n")

    print("Loading medium network...")
    medium_network = Network()
    task.load_model(medium_network, f"{base_dir}/models/task_1/medium/model.pt")
    print("Running medium network bot...")
    results = task.run_network_bot(
        medium_network, network_bot_games, width=16, height=16, mines=40
    )
    print("Finished running medium network bot!")
    print(f"Win Rate: {results['win_rate']}")
    print(f"Average Turns: {results['average_turns']}\n")

    print("Loading hard network...")
    hard_network = Network()
    task.load_model(hard_network, f"{base_dir}/models/task_1/hard/model.pt")
    print("Running hard network bot...")
    results = task.run_network_bot(
        hard_network, network_bot_games, width=30, height=16, mines=99
    )
    print("Finished running hard network bot!")
    print("Finished running medium network bot!")
    print(f"Win Rate: {results['win_rate']}")
    print(f"Average Turns: {results['average_turns']}\n")
