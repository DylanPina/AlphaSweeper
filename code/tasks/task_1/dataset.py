import torch
from torch.utils.data import Dataset


class MineSweeperDataset(Dataset):
    def __init__(self, moves, labels):
        self.moves = moves
        self.labels = labels

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx):
        moves = self.moves[idx]
        label = self.labels[idx]
        return torch.tensor(moves, dtype=torch.float), torch.tensor(
            label, dtype=torch.float
        )
