#!/usr/bin/env python3
"""
Component test: modules/drone.py (DroneKit vehicle + MAVLink)

Read-only telemetry: connect, print mode, GPS, battery, mission next index.
No arming, no mode changes, no motion commands.

Usage (from repo root):
  python3 test/test_drone_link.py
  python3 test/test_drone_link.py --connection udpin:127.0.0.1:14552
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import prepend_import_paths

prepend_import_paths()
import drone


def main():
    parser = argparse.ArgumentParser(description="Test DroneKit link (drone.py)")
    parser.add_argument(
        "--connection",
        default="udpin:127.0.0.1:14552",
        help="MAVLink connection string",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Seconds to print telemetry (Ctrl+C to stop early)",
    )
    parser.add_argument("--baud", type=int, default=None, help="Serial baud if using serial")
    args = parser.parse_args()

    print("Connecting:", args.connection)
    if args.baud is not None:
        vehicle = drone.connect_drone(args.connection, waitready=True, baudrate=args.baud)
    else:
        vehicle = drone.connect_drone(args.connection, waitready=True)

    if vehicle is None:
        print("ERROR: connect failed")
        return 1

    t0 = time.time()
    try:
        while time.time() - t0 < args.duration:
            loc = drone.get_location_relative()
            mode = drone.get_mode()
            armed = drone.vehicle.armed
            batt = drone.get_battery_info()
            nxt = None
            try:
                nxt = drone.get_next_mission_index()
            except Exception as e:
                nxt = "(err: %s)" % e

            print(
                "mode=%s armed=%s | lat=%.7f lon=%.7f alt=%.1f | "
                "next_wp=%s | batt=%s"
                % (mode, armed, loc.lat, loc.lon, loc.alt, nxt, batt)
            )
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        drone.disconnect_drone()
        print("Disconnected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
