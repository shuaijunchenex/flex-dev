from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_utils import Handlers, console

def test_fn_1(matrix_1, matrix_2, norm='l2'):
    print("test_fn_1(matrix_1, matrix_2, norm='l2')")
    print(f"arg1: {matrix_1}")
    print(f"arg2: {matrix_2}")
    print(f"arg3: {norm}")
    return

def test_fn_2(wbab_1, wbab_2, norm='l2'):
    print("test_fn_2(wbab_1, wbab_2, norm='l2')")
    print(f"arg1: {wbab_1}")
    print(f"arg2: {wbab_2}")
    print(f"arg3: {norm}")
    return

def test_handlers():
    console.out(f"Test Handlers")
    console.out("------------- Begin ---------------")
    console.info("Register handlers")

    handlers = Handlers().register_handler("f1", test_fn_1).register_handler("f2", test_fn_2)
    console.info(f"Handler count: {handlers.count}\n")

    console.out("-----------------------------------")
    console.info("Call handler test_fn_1")
    console.info("handlers.invoke_handler('f1', 1, 1, 'a')")
    handlers.invoke_handler("f1", 1, 1, "a")

    args = { "matrix_1": 2, "matrix_2": 2, "norm": "b"}
    console.info("args = { 'matrix_1': 2, 'matrix_2': 2, 'norm': 'b'}")
    console.info("handlers.invoke_handler('f1', **args)")
    handlers.invoke_handler("f1", **args)

    console.info("handlers.invoke_handler('f1', matrix_1=3, matrix_2=3, norm='c')")
    handlers.invoke_handler("f1", matrix_1=3, matrix_2=3, norm="c")

    console.out("-----------------------------------")
    console.info("Call handler test_fn_2")
    console.info("handlers.invoke_handler('f2', 1, 1, 'a')")
    handlers.invoke_handler("f2", 1, 1, "a")

    args = { "wbab_1": 2, "wbab_2": 2, "norm": "b"}
    console.info("args = { 'wbab_1': 2, 'wbab_2': 2, 'norm': 'b'}")
    console.info("handlers.invoke_handler('f2', **args)")
    handlers.invoke_handler("f2", **args)

    console.info("handlers.invoke_handler('f2', wbab_1=3, wbab_2=3, norm='c')")
    handlers.invoke_handler("f2", wbab_1=3, wbab_2=3, norm="c")

    console.out("------------- End -----------------\n")
    return


def main():
    test_handlers()
    return


if __name__ == "__main__":
    main()
