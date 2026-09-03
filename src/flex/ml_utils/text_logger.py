from __future__ import annotations
import os
from datetime import date
from sys import exception

from .string import String


class TextLogger:
    """
    text file log
    """

    def __init__(self, file_extension: str = ".log"):
        """
        Create text logger with log name and store path(default is './log')
        """
        self._log_name: str = ""       # Name of log file
        self._log_path: str = ""       # store log file to path
        self._log_extension = file_extension

        self._stream = None
        self.__date = date.today()
        return

    @property
    def is_open(self) -> bool:
        """
        Indicate logger is open
        """
        if self._stream is None:
            return False
        return not self._stream.closed

    def open(self, log_name: str = "text_log", path: str = "./log"):
        """
        Open text logger with log name(default is 'text_log') and store path(default is './log')
        """
        if String.is_none_or_empty(log_name):
            raise ValueError("Log name can't be empty.")

        self._log_name = log_name       # Name of log file
        self._log_path = path           # store log file to path
        os.makedirs(self._log_path, exist_ok=True)

        self.__date = date.today()
        filename = self.__get_file_name()
        self._stream = open(filename, "a", newline="\n", encoding="utf-8")
        return

    def close(self):
        """
        Close text logger
        """
        if not self.is_open:
            return

        if self._stream is not None:
            self._stream.close()
        self._stream = None
        return

    def write(self, text: str, end = "\n"):
        """
        Write text to logger
        """

        if date.today() != self.__date:     # One day pass, log to new file
            self.close()
            self.open(self._log_name, self._log_path)

        if self._stream is not None:
            self._stream.write(text + end)
        return self

    # private
    def __get_file_name(self) -> str:
        return f"{self._log_path}/{self._log_name}-{self.__date.strftime('%Y%m%d')}{self._log_extension}"

    # destructor
    def __del__(self):
        self.close()
