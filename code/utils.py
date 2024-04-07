from enum import Enum, auto


class UncoverResult(Enum):
    OUT_OF_BOUNDS = auto()
    ALREADY_UNCOVERED = auto()
    MINE = auto()
    OK = auto()
