import logging


def init_logging(log_arg: str = "ERROR") -> None:
    """Initializes logging capabilities for entire applicaiton"""

    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
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
