import os
import json
from common.config import base_dir
from bots.logic_bot.logic_bot_runner import LogicBotRunner


class MineSweeperDataLoader:

    def __init__(self, logger, task: str):
        self.logger = logger
        self.task = task

    def save(self, data, file_name: str):
        """Saves the data to a json file"""

        with open(f"{base_dir}/data/{self.task}/{file_name}.json", "w") as f:
            self.logger.info(
                f"Saving {len(data['board_states'])} board states points to {file_name}.json"
            )
            json.dump(data, f)

    def load(self, file_name: str):
        """Loads the data from a json file"""

        file_path = f"{base_dir}/data/{self.task}/{file_name}.json"
        if not os.path.exists(file_path):
            self.logger.debug(f"Failed to load... {file_path} does not exist")
            ValueError(f"Failed to load... {file_path} does not exist")

        with open(f"{base_dir}/data/task_1/{file_name}.json", "rb") as f:
            data = json.load(f)
            self.logger.info(
                f"Loading {len(data['board_states'])} board states from {file_name}.json"
            )
            return data

    def run_logic_bot(
        self,
        file: str,
        games: int,
        width: int,
        height: int,
        mines: int | None,
        task: str,
    ):
        """Runs the logic bot and saves the results to a json file"""

        runner = LogicBotRunner(games, width, height, mines, task)

        logic_bot_data = runner.run()
        self.save(
            logic_bot_data,
            file,
        )

        return logic_bot_data

    def transform(self, data):
        """Transforms the data into a format that can be used for training"""

        return data
