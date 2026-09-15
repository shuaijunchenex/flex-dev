from __future__ import annotations

import traceback
import sys
from typing import Any, Optional
from colorama import Fore, Back, Style, init

from .string import String
from .text_logger import TextLogger


class console:
    """
    Color support console and log text to file.
    Log level: error > warn > ok > info > out > all
    Notice: call begin_log() to log text to file, end_log() close log file
    """

    __const_default_log_path = "./.log/"
    __const_default_log_level = 4
    __const_char_ok = "[OK] "#"✅ "

    __console_logger: TextLogger = TextLogger()
    __exception_logger: TextLogger = TextLogger()
    __debug_logger: TextLogger = TextLogger()

    __log_level = __const_default_log_level         # Default log level
    __is_debug = True

    __console_log_enabled: bool = False
    __console_log_path = __const_default_log_path
    __console_log_name = "console_trace"

    __exception_log_enabled: bool = False
    __exception_log_path = __const_default_log_path
    __exception_log_name = "exception_trace"

    __debug_log_enabled: bool = False
    __debug_log_path = __const_default_log_path
    __debug_log_name = "debug_trace"

    @staticmethod
    def set_debug(on_off = True):
        """
        Set status if output debug info
        """
        console.__is_debug = on_off

    @staticmethod
    def set_log_level(level_str = "warn"):
        match level_str:
            case "all":
                console.__log_level = 0
            case "out": 
                console.__log_level = 1
            case "info": 
                console.__log_level = 2
            case "ok": 
                console.__log_level = 3
            case "warn": 
                console.__log_level = 4
            case "error": 
                console.__log_level = 5
            case _:
                console.__log_level = 4
        return

    @staticmethod
    def set_console_logger(log_path= "./.log/" , log_name = "console_trace"):
        console.__console_log_path = log_path
        console.__console_log_name = log_name
        return console

    @staticmethod
    def set_exception_logger(log_path="./.log/", log_name = "exception_trace"):
        console.__exception_log_path = log_path
        console.__exception_log_name = log_name
        return console

    @staticmethod
    def set_debug_logger(log_path="./.log/", log_name = "debug_trace"):
        console.__debug_log_path = log_path
        console.__debug_log_name = log_name
        return console

    @staticmethod
    def enable_console_log(enable):
        console.__console_log_enabled = enable
        return console

    @staticmethod
    def enable_exception_log(enable):
        console.__exception_log_enabled = enable
        return console

    @staticmethod
    def enable_debug_log(enable):
        console.__exception_log_enabled = enable
        return console

    #--------------------------------------------------------------
    @staticmethod
    def __begin_log():
        if console.__console_log_enabled and not console.__console_logger.is_open:
            console.__console_logger.open(console.__console_log_name, console.__console_log_path)

        if console.__exception_log_enabled and not console.__exception_logger.is_open:
            console.__exception_logger.open(console.__exception_log_name, console.__exception_log_path)

        if console.__debug_log_enabled and not console.__debug_logger.is_open:
            console.__debug_logger.open(console.__debug_log_name, console.__debug_log_path)
        return    
    
    @staticmethod
    def __end_log():
        console.__console_logger.close()
        console.__exception_logger.close()
        console.__debug_logger.close()
        return
    
    @staticmethod
    def __write_console_log(text: Any, end="\n"):
        console.__begin_log()
        if console.__console_log_enabled:
            console.__console_logger.write(f"{text}", end)
        return

    @staticmethod
    def __write_exception_log(text: Any, end="\n"):
        console.__begin_log()
        if console.__exception_log_enabled:
            console.__exception_logger.write(f"{text}", end)
        return

    @staticmethod
    def __write_debug_log(text: Any, end="\n"):
        console.__begin_log()
        if console.__debug_log_enabled:
            console.__debug_logger.write(f"{text}", end)
        return

    def _supports_utf8() -> bool:
        try:
            return sys.stdout.encoding.lower().startswith("utf")
        except Exception:
            return False

    #--------------------------------------------------------------
    @staticmethod
    def out(text: Any, end = "\n"):
        print(Fore.RESET + f"{text}" + Fore.RESET, end=end)
        if console.__log_level <= 1:
            console.__write_console_log(text)
        return console
    
    @staticmethod
    def info(text: Any, end = "\n"):
        print(Fore.CYAN + f"{text}" + Fore.RESET, end=end)
        if console.__log_level <= 2:
            console.__write_console_log(text)
        return console
    
    @staticmethod
    def ok(text: Any, end = "\n"):
        print(Fore.GREEN + f"{console.__const_char_ok}{text}" + Fore.RESET, end=end)
        if console.__log_level <= 3:
            console.__write_console_log(text)
        return console
    
    @staticmethod
    def warn(text: Any, end = "\n"):
        print(Fore.YELLOW + f"{text}" + Fore.RESET, end=end)
        if console.__log_level <= 4:
            console.__write_console_log(text)
        return console
    
    @staticmethod
    def error(text: Any, end = "\n"):
        print(Fore.RED + f"{text}" + Fore.RESET, end=end)
        if console.__log_level <= 5:
            console.__write_console_log(text)
        return console

    @staticmethod
    def error_exception(text: str, end = "\n"):
        """
        Print error and raise exception, log exception
        """
        s = f"Exception: {text}"
        console.error(s, end)
        console.__write_exception_log(text)
        raise Exception(s)

    @staticmethod
    def raise_exception(ex: Exception, end = "\n"):
        """
        Print error and raise exception, log exception
        """
        if not isinstance(ex, Exception):
            return console.error_exception(f"{ex}", end)

        try:
            console.error(f"Exception: {ex}", end)
            raise ex
        except Exception as e:
            trace_text = traceback.format_exc()
            rep = repr(e)
            arr = [f"{'='*30}",
                   f"Exception message: {e}",
                   f"Exception error:{rep}",
                   f"Exception Trace: {trace_text}",
                   f"{'-'*30}",]

            console.__write_exception_log("\n".join(arr))
            raise

    @staticmethod
    def debug(text: Any, end="\n"):
        """
        Debug output text if debug status is True
        """
        s = f"{text}"
        if console.__is_debug:
            print(Fore.MAGENTA + s + Fore.RESET, end=end)
        if console.__debug_log_enabled:
            console.__write_debug_log(s)
        return console

    #--------------------------------------------------------------
    @staticmethod
    def wait_any_key(prompt_text="Press any key to continue.") -> None:
        """
        Wait any key pressed
        """
        import keyboard
        console.out(prompt_text, end="")
        keyboard.read_key()

    @staticmethod
    def wait_key(key="enter", prompt_text: Optional[str] = None) -> None:
        """
        Wait key pressed
        """
        import keyboard
        if String.is_none_or_empty(prompt_text):
            prompt_text = f"Press {key} to continue."

        console.out(prompt_text, end="")
        keyboard.wait(key)

    @staticmethod
    def __del__():
        console.__end_log()