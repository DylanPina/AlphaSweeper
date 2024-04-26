import sys
import argparse
import json

sys.path.append("../../")


from config import base_dir, init_logging
from logic_bot import LogicBotRunner


class Task1DataLoader:

    def __init__(self):
        pass

    def save(self, data, file_name: str = "task_1_data"):
        """Saves the data to a json file"""

        with open(f"{base_dir}/data/{file_name}.json", "w") as f:
            json.dump(data, f)

    def load(self, file_name: str = "task_1_data"):
        """Loads the data from a json file"""

        with open(f"{base_dir}/data/{file_name}.json", "rb") as f:
            data = json.load(f)

        board_states = data["board_states"]
        revealed_states = data["revealed_states"]
        moves = data["moves"]
        results = data["results"]
        win_rate = data["win_rate"]
        average_turns = data["average_turns"]

        return board_states, revealed_states, moves, results, win_rate, average_turns

    def run_logic_bot(
        self, log_level: str, file: str, games: int, width: int, height: int, mines: int
    ):
        """Runs the logic bot and saves the results to a json file"""

        init_logging(log_level)

        runner = LogicBotRunner(log_level, games, width, height, mines)
        board_states, revealed_states, moves, results, win_rate, average_turns = (
            runner.run()
        )

        print(f"Running logic bot with {games} games")
        print(f"Board: {width}x{height} with {mines} mines")
        print(f"Total games played: {len(moves)}")
        print(f"Win rate: {100 * win_rate:.2f}%")
        print(f"Average turns per game: {average_turns}\n")

        self.save(
            {
                "board_states": board_states,
                "revealed_states": revealed_states,
                "moves": moves,
                "results": results,
                "win_rate": win_rate,
                "average_turns": average_turns,
            },
            file,
        )

        return board_states, moves, results, win_rate, average_turns

    def transform(self, data):
        """Transforms the data into a format that can be used for training"""

        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-log")
    parser.add_argument("-file")
    parser.add_argument("-games")
    parser.add_argument("-width")
    parser.add_argument("-height")
    parser.add_argument("-mines")

    args = parser.parse_args()
    log_level, file, games, width, height, mines = (
        args.log,
        args.file,
        args.games,
        args.width,
        args.height,
        args.mines,
    )

    loader = Task1DataLoader()
    loader.run_logic_bot(
        log_level, file, int(games), int(width), int(height), int(mines)
    )

    board_states, revealed_states, moves, results, win_rate, average_turns = (
        loader.load(file)
    )
