from __future__ import annotations
import sys


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_simu_switcher import SimuSwitcher, SimuNodeDataEventArgs
from flex.ml_simu_switcher import SimuSwitcherDefault

#################################################

def run_switcher():
    switcher.run()

def connect_nodes():
    client1.connect("edge.1")
    client2.connect("edge.1")

    client3.connect("edge.2")
    client4.connect("edge.2")

    edge1.connect("server.1")
    edge2.connect("server.1")


def client_send_up():
    client1.send_up("client-1 up data")
    client2.send_up("client-2 up data")

    client3.send_up("client-3 up data")
    client4.send_up("client-4 up data")

def edge_send_updown():
    edge1.send_down("client.11", "edge-1 down data to Client.11")
    edge2.send_down("client.21", "edge-2 down data to Client.21")

    edge1.send_up("edge-1 up data")
    edge2.send_up("edge-2 up data")

def server_send_down():
    server1.send_down("edge.1", "server-1 down data to edge.1")
    server1.send_down_all("server-1 down data to all")

def on_receive_data(args: SimuNodeDataEventArgs):
    print("Receive event " + args.__str__() + " [" + args.data + "]")
    return

###########################################################

#定义switcher

switcher = SimuSwitcher()
# or
# switcher = SimuSwitcherDefault.switcher

#创建node
client1 = switcher.create_node("client.11")
client1.attach_event("on_receive", on_receive_data)
client2 = switcher.create_node("client.12")
client2.attach_event("on_receive", on_receive_data)
client3 = switcher.create_node("client.21")
client3.attach_event("on_receive", on_receive_data)
client4 = switcher.create_node("client.22")
client4.attach_event("on_receive", on_receive_data)

edge1 = switcher.create_node("edge.1")
edge1.attach_event("on_receive", on_receive_data)
edge2 = switcher.create_node("edge.2")
edge2.attach_event("on_receive", on_receive_data)

server1 = switcher.create_node("server.1")
server1.attach_event("on_receive", on_receive_data)


def main():
    run_switcher()
    connect_nodes()

    client_send_up()
    edge_send_updown()
    server_send_down()


if __name__ == "__main__":
    main()
    print("Press Enter to stop\n")
    sys.stdin.read(1)
    switcher.stop()
    print("switcher stopped")
