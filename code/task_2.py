import argparse
import logging
from common.config import base_dir, configure_logging, setup_logger
from network.network import Network
from common.data_loader import MineSweeperDataLoader
from task import Task


class Task2(Task):

    def __init__(
        self,
    ):
        super().__init__(
            setup_logger("Task 2", task="task_2", log_file="task_2"), task="task_2"
        )
        self.logger = logging.getLogger("Task 2")
        self.data_loader = MineSweeperDataLoader(self.logger, task="task_2")

    def generate_data(self, train_games=10000, test_games=2000):
        """Generates the training data for games with random number of mines"""

        self.logger.info("Loading task 1 network...")
        task1_hard_network = Network()
        task.load_model(task1_hard_network, f"{base_dir}/models/task_1/hard/model.pt")

        self.logger.info(
            "Generating logic bot train data for games with random number of mines..."
        )
        logic_bot_train_data = task.run_logic_bot(
            games=train_games, width=30, height=30
        )
        self.logger.info(
            "Generating logic bot test data for games with random number of mines..."
        )
        logic_bot_test_data = task.run_logic_bot(games=test_games, width=30, height=30)

        self.logger.info(
            "Generating network bot train data for games with random number of mines..."
        )
        network_bot_train_data = task.run_network_bot(
            network=task1_hard_network, games=train_games, width=30, height=30
        )
        network_bot_test_data = task.run_network_bot(
            network=task1_hard_network, games=test_games, width=30, height=30
        )

        self.data_loader.save(logic_bot_train_data, "logic_bot/train")
        self.data_loader.save(logic_bot_test_data, "logic_bot/test")
        self.data_loader.save(network_bot_train_data, "network_bot/train")
        self.data_loader.save(network_bot_test_data, "network_bot/test")

        return {
            "logic_bot_train_data": logic_bot_train_data,
            "logic_bot_test_data": logic_bot_test_data,
            "network_bot_train_data": network_bot_train_data,
            "network_bot_test_data": network_bot_test_data,
        }

    def load_data(self):
        """Loads the training data for games with random number of mines"""

        self.logger.debug("Loading train data...")
        logic_bot_train_data = self.data_loader.load("logic_bot/train")
        logic_bot_test_data = self.data_loader.load("logic_bot/test")
        network_bot_train_data = self.data_loader.load("network_bot/train")
        network_bot_test_data = self.data_loader.load("network_bot/test")

        combined_train_data = {}
        combined_test_data = {}

        def merge_dictionaries(dict1, dict2):
            """Merge two dictionaries. Concatenate lists if keys overlap."""

            for key, value in dict2.items():
                if key in dict1:
                    if isinstance(dict1[key], list) and isinstance(value, list):
                        dict1[key].extend(value)
                    else:
                        self.logger.error(
                            "Data format within dictionaries is not consistent for merging."
                        )
                else:
                    dict1[key] = value

        merge_dictionaries(combined_train_data, logic_bot_train_data)
        merge_dictionaries(combined_train_data, network_bot_train_data)

        merge_dictionaries(combined_test_data, logic_bot_test_data)
        merge_dictionaries(combined_test_data, network_bot_test_data)

        return {
            "train_data": combined_train_data,
            "test_data": combined_test_data,
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

    task = Task2()

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
    results = task.run_network_bot(network, games, width=30, height=30)
    print("Finished running easy network bot!")
    print(f"Win Rate: {results['win_rate']}")
    print(f"Average Turns: {results['average_turns']}\n")
