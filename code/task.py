import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from bots.logic_bot.logic_bot_runner import LogicBotRunner
from common.config import base_dir
from common.dataset import MineSweeperDataset
from network.network import Network
from bots.network_bot.network_bot_runner import NetworkBotRunner
from abc import ABC, abstractmethod


class Task(ABC):

    def __init__(self, logger, task: str):
        self.logger = logger
        self.task = task

    @abstractmethod
    def generate_data(self, train_games=50000, test_games=10000):
        """Generates the training data for easy, medium and hard games"""

        ...

    @abstractmethod
    def load_data(self):
        """Loads the training data for easy, medium and hard games"""

        ...

    def train(
        self,
        network,
        train_data,
        test_data,
        alpha=0.001,
        epochs=10,
        weight_decay=0.0001,
        batch_size=128,
    ):
        """Trains the network"""

        train_dataset = MineSweeperDataset(
            train_data["board_states"], train_data["label_boards"]
        )
        test_dataset = MineSweeperDataset(
            test_data["board_states"], test_data["label_boards"]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {device}")
        network.to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            network.parameters(), lr=alpha, weight_decay=weight_decay
        )

        train_losses, test_losses = [], []
        train_accuracies, test_accuracies = [], []
        elapsed_times = []

        self.logger.info("Starting training...")
        self.logger.info(
            f"# Parameters: {sum(p.numel() for p in network.parameters())}, Epochs: {epochs}, Alpha: {alpha}, Lambda: {weight_decay}, "
            + f"Batch Size: {batch_size} # Train data: {len(train_loader.dataset)}, # Test data: {len(test_loader.dataset)}"
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
            self.logger.info(
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

    def run_network_bot(
        self,
        network: Network,
        games: int,
        width: int,
        height: int,
        mines: int | None = None,
    ):
        """Runs the network bot"""

        runner = NetworkBotRunner(network, games, width, height, mines, task=self.task)
        return runner.run()

    def run_logic_bot(
        self, games: int, width: int, height: int, mines: int | None = None
    ):
        """Runs the logic bot"""

        runner = LogicBotRunner(games, width, height, mines, task=self.task)
        return runner.run()

    def load_model(self, network: Network, file: str):
        """Loads the model from the file"""

        self.logger.info(f"Loading model from {file}...")
        return network.load_state_dict(torch.load(file))

    def save_model(self, network, file: str):
        """Saves the model to the file"""

        self.logger.info(f"Saving model to {file}...")
        torch.save(network.state_dict(), file)

    def plot(
        self,
        train_losses,
        test_losses,
        train_accuracies,
        test_accuracies,
        directory: str,
    ):
        """Plots the loss and accuracy graphs"""

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{base_dir}/graphs/{self.task}/{directory}/loss.png")

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
        plt.savefig(f"{base_dir}/graphs/{self.task}/{directory}/accuracy.png")
