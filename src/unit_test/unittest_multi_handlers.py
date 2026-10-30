from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_utils import MultiHandlers

def fn_1(a, b, c):
    print(f"fn_1, {a}, {b}, {c}")

def fn_2(a, b, c):
    print(f"fn_2, {a}, {b}, {c}")

def fn_3(a, b, c):
    print(f"fn_3, {a}, {b}, {c}")

def fn_4(a, b, c):
    print(f"fn_4, {a}, {b}, {c}")


def main():
    ev = MultiHandlers()

    print("Reg handler: ev-1, handler -> fn_1, fn_2")
    ev.register_handler("ev-1", fn_1).register_handler("ev-1", fn_2)

    print("\nReg handler: ev-2, handler -> fn_3, fn_4")
    ev.register_handler("ev-2", fn_3).register_handler("ev-2", fn_4)

    print("\nInvoke event: ev-1")
    ev.invoke_handler("ev-1", 1,2,3)
    print("\nInvoke event: ev-2")
    ev.invoke_handler("ev-2", 4,5,6)

    print("\nUnregister handler: ev-1, fn_1")
    ev.unregister_handler("ev-1", fn_1)
    print("\nInvoke handler: ev-1")
    ev.invoke_handler("ev-1", 7,8,9)

    print("\nUnregister handler: ev-1, *")
    ev.unregister_handler("ev-1")

    print("\nInvoke handler: ev-1")
    ev.invoke_handler("ev-1", 10,11,12)
    print("\nInvoke handler: ev-2")
    ev.invoke_handler("ev-2", 13,14,15)

if __name__ == "__main__":
    main()
