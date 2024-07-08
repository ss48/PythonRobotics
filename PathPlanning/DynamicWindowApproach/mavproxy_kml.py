#!/usr/bin/env python3
import pexpect
import time

# Define the MAVProxy command
mavproxy_cmd1 = "mavproxy.py --master=udpin:127.0.0.1:14550 --console --map --out=udpout:127.0.0.1:14552"
mavproxy_cmd2 = "mavproxy.py --master=udpin:127.0.0.1:14551 --console --map --out=udpout:127.0.0.1:14553"
# Start MAVProxy
child = pexpect.spawn(mavproxy_cmd1)
child = pexpect.spawn(mavproxy_cmd2)
# Wait for MAVProxy to load
child.expect("Received", timeout=60)

# Load the kmlread module
child.sendline("module load kmlread")

# Wait for the module to load
time.sleep(2)

# Load the KML file
child.sendline("kml load /home/dell/Essex_Hospitals_NoFlyZones.kml")

# Give some time for the KML to load and process
time.sleep(5)

# Optionally, you can add more commands here, e.g., to toggle visibility or create fences

# Interact with MAVProxy console
child.interact()

