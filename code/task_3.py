import argparse
import logging
from common.config import base_dir, configure_logging, setup_logger
from network.network import Network
from common.data_loader import MineSweeperDataLoader
from .task import Task


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

        self.logger.info(
            "Generating logic bot train data for games with random number of mines..."
        )
        train_data = task.run_logic_bot(games=train_games, width=30, height=30)
        self.logger.info(
            "Generating logic bot test data for games with random number of mines..."
        )
        test_data = task.run_logic_bot(games=test_games, width=30, height=30)

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
    parser.add_argument("-network_bot_games")

    args = parser.parse_args()
    (
        log_level,
        train,
        get_data,
        network_bot_games,
    ) = (args.log, args.train, args.get_data, args.network_bot_games)

    configure_logging(log_level)

    task = Task3()

    if train:
        if get_data:
            task.generate_data()

        data = task.load_data()

        network = Network()
        train_data, test_data = (
            data["train_data"],
            data["test_data"],
        )
        train_losses, train_accuracies, test_losses, test_accuracies, elapsed_times = (
            task.train(
                network=network,
                train_data=train_data,
                test_data=test_data,
                alpha=0.001,
                epochs=5,
                weight_decay=0.00001,
            )
        )
        task.save_model(network, f"{base_dir}/models/task_2/model.pt")
        task.plot(
            train_losses,
            test_losses,
            train_accuracies,
            test_accuracies,
            directory="final",
        )

    print("Loading network...")
    network = Network()
    task.load_model(network, f"{base_dir}/models/task_2/model.pt")
    print("Running network bot...")
    results = task.run_network_bot(network, network_bot_games, width=30, height=30)
    print("Finished running easy network bot!")
    print(f"Win Rate: {results['win_rate']}")
    print(f"Average Turns: {results['average_turns']}\n")
