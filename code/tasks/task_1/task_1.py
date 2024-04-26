import argparse
from .data_loader import Task1DataLoader


class Task1:

    def __init__(
        self, log_level: str, file: str, games: int, width: int, height: int, mines: int
    ):
        self.log_level = log_level
        self.file = file
        self.games = games
        self.width = width
        self.height = height
        self.mines = mines
        self.data_loader = Task1DataLoader()
        self.data = self.load()

    def load(self):
        """Loads the data from the json file"""

        data = self.data_loader.load(self.file)
        if not data:
            data = self.data_loader.run_logic_bot(
                self.log_level,
                self.file,
                self.games,
                self.width,
                self.height,
                self.mines,
            )

        return data

    def run(self):
        """Runs the task and saves the results to a json file"""

        pass


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

    task = Task1(log_level, file, games, width, height, mines)
    task.run()
