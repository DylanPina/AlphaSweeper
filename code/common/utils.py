import torch
from enum import Enum, auto


class GameResult(Enum):
    OUT_OF_BOUNDS = auto()
    ALREADY_UNCOVERED = auto()
    MINE = auto()
    OK = auto()
    WIN = auto()


def transform(board):
    """Prepares the board for use in the network"""

    board_mapped = torch.zeros(board.size(), dtype=board.dtype, device=board.device)
    board_mapped[board == -1] = 0
    board_mapped[board == -2] = 1
    clues_mask = (board >= 0) & (board <= 8)
    board_mapped[clues_mask] = board[clues_mask] + 2
    return board_mapped
