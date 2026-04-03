#!/usr/bin/env python3
"""
Component test: modules/lidar.py (serial rangefinder)

Reads distance (and optionally temperature) until Ctrl+C or --count.

Usage:
  python3 test/test_lidar_component.py --port /dev/ttyTHS1
  python3 test/test_lidar_component.py --port /dev/ttyUSB0 --count 20
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import prepend_import_paths

prepend_import_paths()
import lidar


def main():
    parser = argparse.ArgumentParser(description="Test lidar serial reads")
    parser.add_argument("--port", default="/dev/ttyTHS1", help="Serial device path")
    parser.add_argument("--count", type=int, default=0, help="Stop after N reads (0 = infinite)")
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between reads")
    args = parser.parse_args()

    print("Opening lidar:", args.port)
    res = lidar.connect_lidar(args.port)
    print("connect_lidar:", res)
    if not lidar.check_connection():
        print("ERROR: serial not open")
        return 1

    n = 0
    try:
        while args.count == 0 or n < args.count:
            dist, strength = lidar.read_lidar_distance(timeout_s=0.5)
            print("n=%d distance_m=%s strength=%s" % (n, dist, strength))
            n += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        lidar.disconnect_lidar()
        print("Disconnected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
