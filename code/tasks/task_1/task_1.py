from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from common.config import base_dir
from tasks.task_1.dataset import MineSweeperDataset
from tasks.task_1.network import Task1Network
from .data_loader import Task1DataLoader


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
        self.train_data, self.train_labels = self.load(train_data_file, train_games)
        self.test_data, self.test_labels = self.load(test_data_file, test_games)

    def load(self, file: str, games: int):
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

    def train(self, network=Task1Network(), alpha=0.01, epochs=100):
        """Trains the network"""

        train_dataset = MineSweeperDataset(self.train_data, self.train_labels)
        test_dataset = MineSweeperDataset(self.test_data, self.test_labels)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=3, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=3, shuffle=True
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        network.to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(network.parameters(), lr=alpha)

        train_losses, test_losses = [], []
        train_accuracies, test_accuracies = [], []
        elapsed_times = []

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
            print(
                f"Epoch {epoch + 1}/{epochs} completed with average train loss = {avg_train_loss}, average test loss = {avg_test_loss},"
                + f"average train accuracy = {train_accuracies[-1]}, average test accuracy = {test_accuracies[-1]}, time elapsed = {elapsed_times[-1]} s"
            )

        return (
            train_losses,
            train_accuracies,
            test_losses,
            test_accuracies,
            elapsed_times,
        )

    def plot(self, train_losses, test_losses, train_accuracies, test_accuracies):
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
