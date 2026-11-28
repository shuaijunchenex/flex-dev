from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_utils import console


def main():

    console.enable_console_log(True)
    console.out("Console test without log to file")
    console.out("------------- Begin ---------------")
    console.out("Normal text")
    console.info("Info text")
    console.ok("Ok text")
    console.warn("Warn text")
    console.error("Error text")
    console.out("------------- End -----------------")

    console.out("Console test with log to file")
    console.out("------------- Begin ---------------")

    console.set_log_level("warn")

    console.out("Out text with logger")
    console.info("Info text with logger")
    console.ok("Ok text with logger")
    console.warn("Warn text with logger")
    console.error("Error text with logger")
    console.out("------------- End -----------------")

    console.out("\n\nWait any key test")
    console.out("----------------------------------")
    console.wait_any_key()

    console.out("\nWait key test")
    console.out("----------------------------------")
    console.wait_key()
    console.info("ENTER key pressed")

    return

if __name__ == "__main__":
    main()
