import torch
import random
from common.utils import transform
from tasks.task_1.network import Task1Network
from game.minesweeper import Minesweeper


class NetworkBot:
    def __init__(self, network: Task1Network, game: Minesweeper, logger):
        self.network = network
        self.game = game
        self.logger = logger

    def play_turn(self, turn_number: int):
        """Plays a turn and returns the result"""

        row, col = self.get_next_move()
        self.logger.info(f"Turn {turn_number} - Selected cell: ({row}, {col})")

        return self.game.play_turn((row, col))

    def get_next_move(self):
        """Returns the next move"""

        input = torch.tensor([self.game.user_board]).unsqueeze(1)
        input_transformed = transform(input)

        output = self.network(input_transformed)
        output_transformed = output.squeeze()
        output_masked = self.apply_mask(output_transformed, input_transformed.squeeze())

        self.logger.debug(f"Network output:\n{output_masked}")

        next_move = random.choice(
            (output_masked == torch.min(output_masked))
            .nonzero()
            .squeeze()
            .detach()
            .numpy()
        )

        row, col = next_move
        return row, col

    def apply_mask(self, input: torch.Tensor, board: torch.Tensor):
        """Applies a mask to the input tensor which prevents the bot from selecting cells that are set to 1 by assigning them a very high negative value (for minimization problems)."""

        mask = board != 1
        input[mask] = float("inf")
        return input
