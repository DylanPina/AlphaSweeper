import logging
import os
import json
from common.config import base_dir, init_logging
from logic_bot.logic_bot_runner import LogicBotRunner


class Task1DataLoader:

    def __init__(self):
        pass

    def save(self, data, file_name):
        """Saves the data to a json file"""

        with open(f"{base_dir}/data/task_1/{file_name}.json", "w") as f:
            json.dump(data, f)

    def load(self, file_name: str = "task_1_data"):
        """Loads the data from a json file"""

        file_path = f"{base_dir}/data/task_1/{file_name}.json"
        if not os.path.exists(file_path):
            logging.debug(f"{file_path} does not exist")
            return None

        with open(f"{base_dir}/data/task_1/{file_name}.json", "rb") as f:
            logging.info(f"Loading {file_name}.json")
            return json.load(f)

    def run_logic_bot(
        self, log_level: str, file: str, games: int, width: int, height: int, mines: int
    ):
        """Runs the logic bot and saves the results to a json file"""

        init_logging(log_level)

        runner = LogicBotRunner(log_level, games, width, height, mines)
        board_states, revealed_states, moves, results, win_rate, average_turns = (
            runner.run()
        )

        logging.info(f"Running logic bot with {games} games")
        logging.info(f"Board: {width}x{height} with {mines} mines")
        logging.info(f"Total games played: {len(moves)}")
        logging.info(f"Win rate: {100 * win_rate:.2f}%")
        logging.info(f"Average turns per game: {average_turns}\n")

        logic_bot_data = {
            "board_states": board_states,
            "revealed_states": revealed_states,
            "moves": moves,
            "results": results,
            "win_rate": win_rate,
            "average_turns": average_turns,
        }

        self.save(
            logic_bot_data,
            file,
        )

        return logic_bot_data

    def transform(self, data):
        """Transforms the data into a format that can be used for training"""

        return data
