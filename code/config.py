import os
import logging


difficulty_levels = {
    "EASY": (9, 9, 10),
    "MEDIUM": (16, 16, 40),
    "HARD": (30, 16, 99),
}

base_dir = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/final-project/code"


def init_logging(log_arg: str = "ERROR", file_name: str = "log") -> None:
    """Initializes logging capabilities for entire applicaiton"""

    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.debug,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    log_path = os.path.join(base_dir, "logs", f"{file_name}.log")

    log_level = log_levels[log_arg]
    logging.basicConfig(
        level=log_level,
        filename=log_path,
        filemode="w",
        format="%(asctime)s - [%(levelname)s]: %(message)s",
    )
