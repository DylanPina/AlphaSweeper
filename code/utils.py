from enum import Enum, auto


class GameResult(Enum):
    OUT_OF_BOUNDS = auto()
    ALREADY_UNCOVERED = auto()
    MINE = auto()
    OK = auto()
