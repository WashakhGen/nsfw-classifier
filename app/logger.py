import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_FILE = Path(__file__).parent.parent / "logs" / "app.log"


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record = logging.makeLogRecord(record.__dict__)  # copy so the file handler stays uncolored
        record.levelname = f"\033[32m{record.levelname}\033[0m"  # green
        return super().format(record)


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("nsfw")
    if logger.hasHandlers():  # already configured, e.g. on reload
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False  # keep uvicorn's root logger from printing it twice

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(ColorFormatter(LOG_FORMAT))
    logger.addHandler(stream)

    LOG_FILE.parent.mkdir(exist_ok=True)
    file = logging.FileHandler(LOG_FILE, mode="a")
    file.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file)

    return logger


logger = _build_logger()


def log_main(*messages: object) -> None:
    for message in messages:
        logger.info(message, stacklevel=2)
