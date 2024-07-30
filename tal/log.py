from enum import Enum


class LogLevel(Enum):
    """
    LogLevel, from more verbosity to less verbosity.
    """
    TRACE = 0  # extra debug
    DEBUG = 1
    INFO = 2
    PROGRESS = 3  # progress bars
    WARNING = 4
    ERROR = 5
    PROMPT = 6  # only input() required by the user
    NONE = 7  # no log


_log_level = LogLevel.TRACE


def _set_default_log_level():
    from tal.config import ask_for_config, Config
    global _log_level
    _log_level = LogLevel[ask_for_config(Config.LOG_LEVEL, force_ask=False)]


def set_log_level(level: LogLevel):
    """
    Controls the amount of information that will be logged to the console.

    See tal.LogLevel for the different levels of verbosity.
    """
    global _log_level
    _log_level = level
    from tal.config import Config, read_config, write_config
    config = read_config()
    config[Config.LOG_LEVEL.value[0]] = level.name
    write_config(config)


def log(level: LogLevel, message: str, **kwargs):
    """
    Logs a message to the console.
    """
    import sys
    if level.value >= _log_level.value:
        # TODO add pretty colors (see libcpp-common) :^)
        print(message, **kwargs, file=sys.__stdout__)


def TQDMLogRedirect():
    import sys
    if LogLevel.PROGRESS.value >= _log_level.value:
        return sys.__stderr__
    else:
        return EmptyLogRedirect()


class EmptyLogRedirect:
    def __init__(self):
        pass

    def write(self, text):
        pass

    def flush(self):
        pass

    def getvalue(self):
        return ''
