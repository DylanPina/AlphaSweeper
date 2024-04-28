import torch
import logging
from tasks.task_1.network import Task1Network
from game.minesweeper import Minesweeper


class NetworkBot:
    def __init__(self, network: Task1Network, game: Minesweeper):
        self.network = network
        self.game = game

    def play_turn(self, turn_number: int):
        """Plays a turn and returns the result"""

        row, col = self.get_next_move()
        logging.debug(f"Turn {turn_number} - Selected cell: ({row}, {col})")

        return self.game.play_turn((row, col))

    def get_next_move(self):
        """Returns the next move"""

        current_board_tensor = torch.tensor([self.game.user_board]).float().unsqueeze(0)
        output = self.network(current_board_tensor)
        output_transformed = output.squeeze()
        output_masked = self.apply_mask(
            output_transformed, current_board_tensor.squeeze()
        )
        next_move = (
            (output_masked == torch.max(output_masked))
            .nonzero()
            .squeeze()
            .detach()
            .numpy()
        )
        row, col = next_move
        return row, col

    def apply_mask(self, input: torch.Tensor, board: torch.Tensor):
        """Applies a mask to the input tensor which prevents the bot from uncovering cells that have already been revealed"""

        mask = (board == -2).float()
        return input * mask
