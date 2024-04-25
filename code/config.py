import logging

difficulty_levels = {
    "EASY": (9, 9, 10),
    "MEDIUM": (16, 16, 40),
    "HARD": (30, 16, 99),
}


def init_logging(log_arg: str = "ERROR") -> None:
    """Initializes logging capabilities for entire applicaiton"""

    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.debug,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    log_level = log_levels[log_arg]
    logging.basicConfig(
        level=log_level,
        filename="logs/log.log",
        filemode="w",
        format="%(asctime)s - [%(levelname)s]: %(message)s",
    )
