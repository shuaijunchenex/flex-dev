# ml-simu-switcher Package

The 'ml-simu-switcher' package simulate exchange between nodes.

## Usage

  - Step 1: Create a switcher

```python
# create data exchange simulate switcher
switcher = SimuSwitcher()
```

  - Step 2: Create nodes

```python
# define event callback function
def on_receive_data(args: SimuNodeEventArgs):
    print("Receive event " + args.__str__() + " [" + args.data + "]")
    return

# create client nodes and attach event callback
client1 = switcher.create_node("client.11")			   #node id: "client.11"
client1.attach_event("on_receive", on_receive_data)	    #attach event callback handler
client2 = switcher.create_node("client.12")
client2.attach_event("on_receive", on_receive_data)
client3 = switcher.create_node("client.21")
client3.attach_event("on_receive", on_receive_data)
client4 = switcher.create_node("client.22")
client4.attach_event("on_receive", on_receive_data)

#create edge nodes
edge1 = switcher.create_node("edge.1")
edge1.attach_event("on_receive", on_receive_data)
edge2 = switcher.create_node("edge.2")
edge2.attach_event("on_receive", on_receive_data)

#create server node
server1 = switcher.create_node("server.1")
server1.attach_event("on_receive", on_receive_data)
```

  - Step 3: Connect nodes

```python
# Nodes structure
#  server.1 
#    |- edge.1
#    |    |- client.11
#    |    |- client.12
#    |- edge.2
#         |-client.21
#         |-client.22

client1.connect("edge.1")
client2.connect("edge.1")

client3.connect("edge.2")
client4.connect("edge.2")

edge1.connect("server.1")
edge2.connect("server.1")
```

- Step 4: Exchange data
```python
# client send data up
client1.send_up("client-1 up data to edge.1")
client2.send_up("client-2 up data to edge.1")

client3.send_up("client-3 up data to edge.2")
client4.send_up("client-4 up data to edge.2")

# edge send data up and down
edge1.send_down("client.11", "edge-1 down data to Client.11")
edge2.send_down("client.21", "edge-2 down data to Client.21")

edge1.send_up("edge-1 up data to server.1")
edge2.send_up("edge-2 up data to server.1")

# server send data down
server1.send_down("edge.1", "server.1 down data to edge.1")
server1.send_down_all("server.1 down data to all via edge.1 and edge.2")
```




## Package build & deploy

- Build package command

```cmd
python -m build
```
- Upload package to Nexus command

```cmd
twine upload -r nexus dist/*
```

- Ruff check & format code
```cmd
ruff check
ruff format
ruff check --fix
```