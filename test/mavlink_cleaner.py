#!/usr/bin/env python3
from pymavlink import mavutil

INP  = 'udpin:127.0.0.1:14551'     # listen here (from MAVProxy)
OUTP = 'udpout:127.0.0.1:14552'    # forward clean stream here

inp = mavutil.mavlink_connection(INP)
out = mavutil.mavlink_connection(OUTP)

print("Forwarding MAVLink 14551 -> 14552 (dropping ADSB heartbeats)...")

while True:
    msg = inp.recv_msg()
    if msg is None:
        continue

    # Drop only the ADSB heartbeat that breaks DroneKit:
    # (autopilot=8 MAV_AUTOPILOT_INVALID, type=27 MAV_TYPE_ADSB)
    if msg.get_type() == 'HEARTBEAT':
        try:
            if int(msg.autopilot) == 8 and int(msg.type) == 27:
                continue
        except Exception:
            pass

    out.mav.send(msg)
