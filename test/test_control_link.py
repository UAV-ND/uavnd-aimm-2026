#!/usr/bin/env python3
"""
Component test: modules/control.py (PID config + wrappers over drone)

Connects, configures PID like the mission, prints mode / pilot_took_over /
next mission index / flight altitude. No arming requirement for read-only
checks (connect may still need a running MAVLink endpoint).

Usage:
  python3 test/test_control_link.py
  python3 test/test_control_link.py --connection udpin:127.0.0.1:14552
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import prepend_import_paths

prepend_import_paths()
import control


def main():
    parser = argparse.ArgumentParser(description="Test control.py wrappers")
    parser.add_argument("--connection", default="udpin:127.0.0.1:14552")
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()

    print("Connecting via control.connect_drone...")
    control.connect_drone(args.connection)

    control.configure_PID("PID")
    control.set_flight_altitude(4.0)

    print("PID configured; flight_altitude set to %.1f" % control.get_flight_altitude())

    t0 = time.time()
    try:
        while time.time() - t0 < args.duration:
            mode = control.get_mode()
            po = control.pilot_took_over()
            nxt = control.get_next_mission_index()
            alt = control.get_flight_altitude()
            print("mode=%s pilot_took_over=%s next_idx=%s cmd_alt=%.1f" % (mode, po, nxt, alt))
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        try:
            control.stop_drone()
        except Exception:
            pass
        import drone
        drone.disconnect_drone()
        print("Disconnected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
