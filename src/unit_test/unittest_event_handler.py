from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_utils import EventArgs, EventHandler

def on_event_1(args: EventArgs):
    print("raise event_1, AAA")

def on_event_2(args: EventArgs):
    print("raise event_2, BBB")

def on_event_3(args: EventArgs):
    print("raise event_3, CCC")

def on_event_4(args: EventArgs):
    print("raise event_4, DDD")


def main():
    ev = EventHandler()

    ev.attach_event("ev-1", on_event_1)
    print("attach event: ev-1, handler -> on_event_1")
    ev.attach_event("ev-1", on_event_2)
    print("attach event: ev-1, handler -> on_event_2")

    ev.attach_event("ev-2", on_event_3)
    print("attach event: ev-2, handler -> on_event_3")
    ev.attach_event("ev-2", on_event_4)
    print("attach event: ev-2, handler -> on_event_4")

    print("raise event: ev-1")
    ev.raise_event("ev-1", EventArgs())
    print("raise event: ev-2")
    ev.raise_event("ev-2", EventArgs())

    print("detach event: ev-1, on_event_1")
    ev.detach_event("ev-1", on_event_1)
    print("raise event: ev-1")
    ev.raise_event("ev-1", EventArgs())

    print("detach event: ev-1")
    ev.detach_event("ev-1")

    print("raise event: ev-1")
    ev.raise_event("ev-1", EventArgs())
    print("raise event: ev-2")
    ev.raise_event("ev-2", EventArgs())

if __name__ == "__main__":
    main()
