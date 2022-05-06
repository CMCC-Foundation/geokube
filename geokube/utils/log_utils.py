import logging
import logging.handlers
from typing import Any, Optional


class LogObject:
    def __init__(
        self,
        username: Optional[str] = "<unknown>",
        request_id: Optional[str] = "<unknown>",
    ):
        self.username = username
        self.request_id = request_id

    @classmethod
    def new(cls):
        return LogObject()

    @property
    def dict_repr(self):
        return {"user_id": self.username, "request_id": self.request_id}

    def __repr__(self):
        return str(self.dict_repr)

    def __str__(self):
        return str(self.__repr__())


class SystemLogFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "user_id"):
            record.user_id = "<unknown>"
        if not hasattr(record, "request_id"):
            record.request_id = "<unknown>"
        return True


def configure_logger(
    logger_name: str,
    log_path: str = "./geokube",
    debug_level: Any = logging.DEBUG,
) -> logging.Logger:
    logger = logging.getLogger(logger_name)

    # if logger.hasHandlers():
    #     return logger

    if not log_path.endswith("log"):
        log_path = log_path + ".log"
    formatter = logging.Formatter(
        "%(asctime)s:%(msecs)s %(levelname)s %(user_id)s %(request_id)s"
        " %(module)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger.setLevel(debug_level)
    fh = logging.handlers.RotatingFileHandler(log_path, "a", 30000, 30)
    fh.setLevel(debug_level)

    ch = logging.StreamHandler()
    ch.setLevel(debug_level)

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addFilter(SystemLogFilter())
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
