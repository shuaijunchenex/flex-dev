from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_utils import TextLogger


def main():

    print("Create text logger")
    text_logger = TextLogger()
    print(f"logger is open: {text_logger.is_open}")

    print("Open logger")
    text_logger.open("test_text_logger")
    print(f"logger is open: {text_logger.is_open}")

    print("Write sonme text")
    text_logger.write("some text")
    text_logger.write("some text", "")
    text_logger.write(" not finish", "")

    print("Close logger")
    text_logger.close()
    print(f"logger is open: {text_logger.is_open}")
    return

if __name__ == "__main__":
    main()
