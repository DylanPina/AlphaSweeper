import argparse
import logging
from common.config import base_dir, configure_logging, setup_logger
from network.network import Network
from common.data_loader import MineSweeperDataLoader
from task import Task


class Task3(Task):

    def __init__(
        self,
    ):
        super().__init__(
            setup_logger("Task 3", task="task_3", log_file="task_3"), task="task_3"
        )
        self.logger = logging.getLogger("Task 3")
        self.data_loader = MineSweeperDataLoader(self.logger, task="task_3")

    def generate_data(self, train_games=10000, test_games=2000):
        """Generates the training data for games with different board of mines"""

        self.logger.info("Generating logic bot train data for 10x10 boards...")
        train_logic_10 = task.run_logic_bot(games=train_games, width=10, height=30)
        self.logger.info("Generating network bot train data for 10x10 boards...")
        test_logic_10 = task.run_logic_bot(games=test_games, width=10, height=10)

        self.logger.info("Generating logic bot train data for 20x20 boards...")
        train_logic_20 = task.run_logic_bot(games=train_games, width=20, height=20)
        self.logger.info("Generating network bot train data for 20x20 boards...")
        test_logic_20 = task.run_logic_bot(games=test_games, width=20, height=20)

        self.logger.info("Generating logic bot train data for 30x30 boards...")
        train_logic_30 = task.run_logic_bot(games=train_games, width=30, height=30)
        self.logger.info("Generating network bot train data for 30x30 boards...")
        test_logic_30 = task.run_logic_bot(games=test_games, width=30, height=30)

        self.logger.info("Generating logic bot train data for 40x40 boards...")
        train_logic_40 = task.run_logic_bot(games=train_games, width=40, height=40)
        self.logger.info("Generating network bot train data for 40x40 boards...")
        test_logic_40 = task.run_logic_bot(games=test_games, width=40, height=40)

        self.logger.info("Generating logic bot train data for 50x50 boards...")
        train_logic_50 = task.run_logic_bot(games=train_games, width=50, height=50)
        self.logger.info("Generating network bot train data for 50x50 boards...")
        test_logic_50 = task.run_logic_bot(games=test_games, width=50, height=50)

        self.logger.info("Loading task 2 network...")
        task2_network = Network()
        task.load_model(task2_network, f"{base_dir}/models/task_2/model.pt")

        self.logger.info("Generating network bot train data for 10x10 boards...")
        train_network_10 = task.run_network_bot(
            games=train_games, width=10, height=10, network=task2_network
        )
        self.logger.info("Generating network bot test data for 10x10 boards...")
        test_network_10 = task.run_network_bot(
            games=test_games, width=10, height=10, network=task2_network
        )

        self.logger.info("Generating network bot train data for 20x20 boards...")
        train_network_20 = task.run_network_bot(
            games=train_games, width=20, height=20, network=task2_network
        )
        self.logger.info("Generating network bot test data for 20x20 boards...")
        test_network_20 = task.run_network_bot(
            games=test_games, width=20, height=20, network=task2_network
        )

        self.logger.info("Generating network bot train data for 30x30 boards...")
        train_network_30 = task.run_network_bot(
            games=train_games, width=30, height=30, network=task2_network
        )
        self.logger.info("Generating network bot test data for 30x30 boards...")
        test_network_30 = task.run_network_bot(
            games=test_games, width=30, height=30, network=task2_network
        )

        self.logger.info("Generating network bot train data for 40x40 boards...")
        train_network_40 = task.run_network_bot(
            games=train_games, width=40, height=40, network=task2_network
        )
        self.logger.info("Generating network bot test data for 40x40 boards...")
        test_network_40 = task.run_network_bot(
            games=test_games, width=40, height=40, network=task2_network
        )

        self.logger.info("Generating network bot train data for 50x50 boards...")
        train_network_50 = task.run_network_bot(
            games=train_games, width=50, height=50, network=task2_network
        )
        self.logger.info("Generating network bot test data for 50x50 boards...")
        test_network_50 = task.run_network_bot(
            games=test_games, width=50, height=50, network=task2_network
        )

        train_data = {
            "logic_10": train_logic_10,
            "logic_20": train_logic_20,
            "logic_30": train_logic_30,
            "logic_40": train_logic_40,
            "logic_50": train_logic_50,
            "network_10": train_network_10,
            "network_20": train_network_20,
            "network_30": train_network_30,
            "network_40": train_network_40,
            "network_50": train_network_50,
        }

        test_data = {
            "logic_10": test_logic_10,
            "logic_20": test_logic_20,
            "logic_30": test_logic_30,
            "logic_40": test_logic_40,
            "logic_50": test_logic_50,
            "network_10": test_network_10,
            "network_20": test_network_20,
            "network_30": test_network_30,
            "network_40": test_network_40,
            "network_50": test_network_50,
        }

        self.data_loader.save(train_data, "train")
        self.data_loader.save(test_data, "test")

        return {
            "train_data": train_data,
            "test_data": test_data,
        }

    def load_data(self):
        """Loads the training data for games with random number of mines"""

        self.logger.debug("Loading train data...")
        train_data = self.data_loader.load("train")
        test_data = self.data_loader.load("test")

        return {
            "train_data": train_data,
            "test_data": test_data,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-log")
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-get_data", action="store_true")
    parser.add_argument("-games")

    args = parser.parse_args()
    (
        log_level,
        train,
        get_data,
        games,
    ) = (args.log, args.train, args.get_data, args.games)

    configure_logging(log_level)

    task = Task3()

    if train:
        if get_data:
            task.generate_data()

        data = task.load_data()

        network = Network()
        train_losses, test_losses = [], []
        train_accuracies, test_accuracies = [], []
        elapsed_times = []

        for train_data, test_data in zip(
            data["train_data"].values(), data["test_data"].values()
        ):
            (
                train_losses,
                train_accuracies,
                test_losses,
                test_accuracies,
                elapsed_times,
            ) = task.train(
                network=network,
                train_data=train_data,
                test_data=test_data,
                alpha=0.001,
                epochs=3,
                weight_decay=0.00001,
            )
        task.save_model(network, f"{base_dir}/models/task_3/model.pt")
        task.plot(
            train_losses,
            test_losses,
            train_accuracies,
            test_accuracies,
            directory="final",
        )

    print("Loading network...")
    network = Network()
    task.load_model(network, f"{base_dir}/models/task_3/model.pt")

    for board_size in range(10, 50, 5):
        print(f"Running network bot for board size: {board_size}")
        network_bot_results = task.run_network_bot(
            network, games, width=board_size, height=board_size
        )
        print("Finished running network bot!")
        print(f"Win Rate: {network_bot_results['win_rate']}")
        print(f"Average Turns: {network_bot_results['average_turns']}\n")

        print(f"Running logic bot for board size: {board_size}")
        logic_bot_results = task.run_logic_bot(
            games, width=board_size, height=board_size
        )
        print("Finished running logic bot!")
        print(f"Win Rate: {logic_bot_results['win_rate']}")
        print(f"Average Turns: {logic_bot_results['average_turns']}\n")
