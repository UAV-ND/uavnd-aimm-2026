#!/usr/bin/env python3
"""
Component test: boat_radio.py (NAMED_VALUE_FLOAT BOAT_LAT / BOAT_LON)

Connects the same way as the mission, registers the listener, polls
read_boat_gps(). Requires MAVLink traffic with those named floats (e.g. from
SITL injection or real FC + boat_gps.lua).

Usage:
  python3 test/test_boat_radio_component.py
  python3 test/test_boat_radio_component.py --connection udpin:127.0.0.1:14552
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import prepend_import_paths

prepend_import_paths()
import drone
import boat_radio


def main():
    parser = argparse.ArgumentParser(description="Test boat GPS MAVLink listener")
    parser.add_argument("--connection", default="udpin:127.0.0.1:14552")
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    print("Connecting:", args.connection)
    drone.connect_drone(args.connection, waitready=True)
    boat_radio.connect_boat_radio(drone.vehicle)

    t0 = time.time()
    try:
        while time.time() - t0 < args.duration:
            fix = boat_radio.read_boat_gps()
            print("boat_gps =", fix)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        boat_radio.disconnect_boat_radio()
        drone.disconnect_drone()
        print("Disconnected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
