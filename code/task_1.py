import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import argparse
from common.config import base_dir, configure_logging, setup_logger
from tasks.task_1.dataset import MineSweeperDataset
from tasks.task_1.network import Task1Network
from tasks.task_1.data_loader import Task1DataLoader
from bots.network_bot.network_bot_runner import NetworkBotRunner


class Task1:

    def __init__(
        self,
        log_level: str,
        test_data_file: str,
        train_data_file: str,
        test_games: int,
        train_games: int,
        width: int,
        height: int,
        mines: int,
    ):
        self.log_level = log_level
        self.test_data_file = test_data_file
        self.train_data_file = train_data_file
        self.train_games = train_games
        self.test_games = test_games
        self.width = width
        self.height = height
        self.mines = mines
        self.network = Task1Network()
        self.data_loader = Task1DataLoader()
        self.train_data, self.train_labels = self.load_data(
            train_data_file, train_games
        )
        self.test_data, self.test_labels = self.load_data(test_data_file, test_games)

    def train(self, network=Task1Network(), alpha=0.001, epochs=10, weight_decay=1e-4):
        """Trains the network"""

        logger = setup_logger("Task 1 Training", f"{base_dir}/logs/task_1/task_1.log")

        train_dataset = MineSweeperDataset(self.train_data, self.train_labels)
        test_dataset = MineSweeperDataset(self.test_data, self.test_labels)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=False
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        network.to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            network.parameters(), lr=alpha, weight_decay=weight_decay
        )

        train_losses, test_losses = [], []
        train_accuracies, test_accuracies = [], []
        elapsed_times = []

        logger.info("Starting training for Task 1...")
        logger.info(
            f"# Parameters: {sum(p.numel() for p in network.parameters())}, Epochs: {epochs}, Alpha: {alpha}"
            + f" # Train data: {len(test_loader.dataset)}, # Test data: {len(test_loader.dataset)}"
        )

        network.train()
        for epoch in range(epochs):
            start_time = time.time()

            total_train_loss = total_train_correct = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device).unsqueeze(1), labels.to(
                    device
                ).unsqueeze(1)

                optimizer.zero_grad()
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * labels.size(0)
                predicted = (outputs > 0.5).float()
                total_train_correct += (predicted == labels).float().sum()

            avg_train_loss = total_train_loss / len(train_loader.dataset)
            train_losses.append(avg_train_loss)
            train_accuracies.append(total_train_correct / len(train_loader.dataset))

            network.eval()
            total_test_loss = total_test_correct = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device).unsqueeze(1), labels.to(
                        device
                    ).unsqueeze(1)
                    outputs = network(inputs)
                    loss = criterion(outputs, labels)
                    total_test_loss += loss.item() * labels.size(0)
                    predicted = (outputs > 0.5).float()
                    total_test_correct += (predicted == labels).float().sum()

            avg_test_loss = total_test_loss / len(test_loader.dataset)
            test_losses.append(avg_test_loss)
            test_accuracies.append(total_test_correct / len(test_loader.dataset))

            elapsed_times.append(time.time() - start_time)
            logger.info(
                f"Epoch {epoch + 1}/{epochs} completed with average train loss = {avg_train_loss}, average test loss = {avg_test_loss},"
                + f" average train accuracy = {train_accuracies[-1]}, average test accuracy = {test_accuracies[-1]}, time elapsed = {elapsed_times[-1]} seconds"
            )

        return (
            train_losses,
            train_accuracies,
            test_losses,
            test_accuracies,
            elapsed_times,
        )

    def load_data(self, file: str, games: int):
        """Loads the data from the json file"""

        data = self.data_loader.load(file)
        if not data:
            data = self.data_loader.run_logic_bot(
                self.log_level,
                file,
                games,
                self.width,
                self.height,
                self.mines,
            )

        input = data["board_states"]
        labels = data["revealed_states"]

        return input, labels

    def load_model(self, file: str):
        """Loads the model from the file"""

        return self.network.load_state_dict(torch.load(file))

    def save_model(self, file: str):
        """Saves the model to the file"""

        torch.save(self.network.state_dict(), file)

    def plot(self, train_losses, test_losses, train_accuracies, test_accuracies):
        """Plots the loss and accuracy graphs"""

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{base_dir}/graphs/task_1/loss.png")

        plt.figure(figsize=(10, 5))
        plt.plot(
            range(1, len(train_accuracies) + 1),
            train_accuracies,
            label="Train Accuracy",
        )
        plt.plot(
            range(1, len(test_accuracies) + 1), test_accuracies, label="Test Accuracy"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"{base_dir}/graphs/task_1/accuracy.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-log")
    parser.add_argument("-train", action="store_true")
    parser.add_argument("-train_data_file")
    parser.add_argument("-test_data_file")
    parser.add_argument("-train_games")
    parser.add_argument("-test_games")
    parser.add_argument("-network_bot_games")
    parser.add_argument("-width")
    parser.add_argument("-height")
    parser.add_argument("-mines")

    args = parser.parse_args()
    (
        log_level,
        train,
        test_data_file,
        train_data_file,
        train_games,
        test_games,
        network_bot_games,
        width,
        height,
        mines,
    ) = (
        args.log,
        args.train,
        args.train_data_file,
        args.test_data_file,
        args.train_games,
        args.test_games,
        args.network_bot_games,
        args.width,
        args.height,
        args.mines,
    )

    configure_logging(log_level)

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

    if train:
        train_losses, train_accuracies, test_losses, test_accuracies, elapsed_times = (
            task.train()
        )
        task.save_model(f"{base_dir}/models/task_1/model.pt")
        task.plot(train_losses, test_losses, train_accuracies, test_accuracies)

    network = Task1Network()
    network.load_state_dict(torch.load(f"{base_dir}/models/task_1/model.pt"))

    runner = NetworkBotRunner(
        network=network,
        games=network_bot_games,
        width=width,
        height=height,
        mines=mines,
    )
    runner.run()