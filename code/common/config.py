import os
import logging


base_dir = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/code"


def configure_logging(log_level: str):
    """Sets the log level"""

    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    logging.basicConfig(level=log_levels[log_level], handlers=[logging.NullHandler()])


def setup_logger(name: str, task: str, log_file: str):
    """Returns a configured logger"""

    handler = logging.FileHandler(f"{base_dir}/logs/{task}/{log_file}.log", mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger = logging.getLogger(name)
    logger.addHandler(handler)

    return logger


def close_logger(logger):
    """Closes the logger handler"""

    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
