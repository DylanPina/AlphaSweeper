import torch
from torch.utils.data import Dataset
from common.utils import transform


class MineSweeperDataset(Dataset):
    def __init__(self, moves, labels):
        self.moves = [transform(torch.tensor(move, dtype=torch.int)) for move in moves]
        self.labels = [torch.tensor(label, dtype=torch.float) for label in labels]

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx):
        moves = self.moves[idx]
        label = self.labels[idx]
        return moves, label
