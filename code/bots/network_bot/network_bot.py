import torch
from common.utils import transform
from network.network import Network
from game.minesweeper import Minesweeper


class NetworkBot:
    def __init__(self, network: Network, game: Minesweeper, logger):
        self.network = network
        self.game = game
        self.logger = logger

    def play_turn(self, turn_number: int):
        """Plays a turn and returns the result"""

        row, col = self.get_next_move(turn_number)
        self.logger.info(f"Turn {turn_number + 1} - Selected cell: ({row}, {col})")

        return self.game.play_turn((row, col))

    def get_next_move(self, turn_number: int):
        """Returns the next move"""

        if turn_number == 0:
            return (self.game.height // 2, self.game.width // 2)

        input = torch.tensor([self.game.user_board]).unsqueeze(1)
        input_transformed = transform(input)

        output = self.network(input_transformed)
        output_transformed = output.squeeze()
        output_masked = self.apply_mask(output_transformed, input_transformed.squeeze())

        self.logger.debug("Network output:")
        network_output_string = "\n"
        for row in output_masked.tolist():
            network_output_string += str(row) + "\n"
        self.logger.debug(network_output_string)

        min_index = torch.argmin(output_masked).item()
        row, col = divmod(min_index, output_masked.size(1))

        return int(row), int(col)

    def apply_mask(self, input: torch.Tensor, board: torch.Tensor):
        """
        Applies a mask to the input tensor which prevents the bot from selecting cells that are set to 1
        by assigning them a very high negative value (for minimization problems).
        """

        mask = board != 1
        input[mask] = float("inf")
        return input
