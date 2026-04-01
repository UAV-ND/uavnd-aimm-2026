#!/usr/bin/env python3
import sys
import time
import logging
import traceback
from collections import deque

sys.path.insert(1, '/home/uav-nano/Documents/aimm-dev/uavnd-aimm-2026')
sys.path.insert(1, '/home/uav-nano/Documents/aimm-dev/uavnd-aimm-2026/modules')

import control
import lidar
from target_detector_adapter import TargetDetector

# =========================
# CONFIG
# =========================
UDP_CONNECTION = 'udpin:127.0.0.1:14552'
LIDAR_PORT = '/dev/ttyTHS1'

TAKEOFF_ALT = 2.0
MISSION_GROUNDSPEED = 1.5

RC_START_CHANNEL = '6'
RC_START_THRESHOLD = 1800

MISSION_WAYPOINT_MODE = "first_nav"
# options:
# "first_nav"
# "second_nav"
# "by_index"
MISSION_WAYPOINT_INDEX = 1

TARGET_WAYPOINT_LAT = None
TARGET_WAYPOINT_LON = None
TARGET_WAYPOINT_ALT = None

CENTER_TOL_X_PX = 25
CENTER_TOL_Y_PX = 25
MIN_CONFIDENCE = 0.95
STABLE_CONFIRM_FRAMES = 10

SEARCH_TIMEOUT_S = 60
CENTER_TIMEOUT_S = 60

MA_X_LEN = 5
MA_Y_LEN = 5

RTL_WAIT_BEFORE_EXIT_S = 10

log_path = '/home/uav-nano/Documents/aimm-dev/uavnd-aimm-2026/test/mission_mvp.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger()

STATE = "setup"


def setup():
    log.info("=== mission_mvp.py started ===")

    log.info("Connecting to drone...")
    control.connect_drone(UDP_CONNECTION)
    control.configure_PID("PID")
    control.set_flight_altitude(TAKEOFF_ALT)

    log.info("Connecting lidar...")
    lidar.connect_lidar(LIDAR_PORT)

    log.info("Starting detector...")
    detector = TargetDetector()

    return detector


def load_target_waypoint_from_fc():
    global TARGET_WAYPOINT_LAT, TARGET_WAYPOINT_LON, TARGET_WAYPOINT_ALT

    log.info("Downloading mission from FC / Mission Planner upload...")
    wp = control.get_target_waypoint_from_mission(
        use_waypoint_mode=MISSION_WAYPOINT_MODE,
        explicit_index=MISSION_WAYPOINT_INDEX
    )

    TARGET_WAYPOINT_LAT = wp["lat"]
    TARGET_WAYPOINT_LON = wp["lon"]
    TARGET_WAYPOINT_ALT = TAKEOFF_ALT if wp["alt"] is None or wp["alt"] <= 0 else wp["alt"]

    log.info(
        "Loaded target waypoint from mission: seq=%s lat=%s lon=%s alt=%s",
        wp["seq"], TARGET_WAYPOINT_LAT, TARGET_WAYPOINT_LON, TARGET_WAYPOINT_ALT
    )


def wait_for_arm_and_start():
    log.info("Waiting for RC arm...")
    control.wait_until_armed()

    time.sleep(2.0)

    log.info("Reading target waypoint from uploaded mission")
    load_target_waypoint_from_fc()

    log.info("Waiting for RC start switch on channel %s...", RC_START_CHANNEL)
    control.wait_for_rc_start(channel=RC_START_CHANNEL, threshold=RC_START_THRESHOLD)


def takeoff():
    if control.pilot_took_over():
        return "manual_override"

    log.info("Taking off to %.2f m", TAKEOFF_ALT)
    control.arm_and_takeoff(TAKEOFF_ALT)
    time.sleep(3.0)
    return "goto_target"


def goto_target():
    if control.pilot_took_over():
        return "manual_override"

    log.info("Going to target area waypoint")
    reached = control.goto_gps_location(
        TARGET_WAYPOINT_LAT,
        TARGET_WAYPOINT_LON,
        TARGET_WAYPOINT_ALT,
        groundspeed=MISSION_GROUNDSPEED,
        radius_m=1.5,
        timeout_s=120
    )

    if control.pilot_took_over():
        return "manual_override"

    if not reached:
        log.info("Goto interrupted or timed out, switching to RTL")
        return "rtl"

    control.hold_position(2.0)

    if control.pilot_took_over():
        return "manual_override"

    return "search_target"


def search_target(detector):
    log.info("Searching for target...")
    start = time.time()

    while time.time() - start < SEARCH_TIMEOUT_S:
        if control.pilot_took_over():
            return "manual_override"

        det = detector.get_target_info()
        if det is not None and det["has_target"]:
            log.info("Target detected: conf=%.3f stable_counter=%d",
                     det["confidence"], det["stable_counter"])
            return "center_target"

        control.stop_drone()
        time.sleep(0.05)

    log.info("Search timeout")
    return "rtl"


def center_target(detector):
    log.info("Centering over target...")

    ma_x = deque(maxlen=MA_X_LEN)
    ma_y = deque(maxlen=MA_Y_LEN)

    stable_drop_counter = 0
    start = time.time()

    while time.time() - start < CENTER_TIMEOUT_S:
        if control.pilot_took_over():
            return "manual_override"

        det = detector.get_target_info()

        if det is None or not det["has_target"]:
            log.info("Lost target, returning to search")
            control.stop_drone()
            return "search_target"

        x_error_px = det["x_error_px"]
        y_error_px = det["y_error_px"]
        conf = det["confidence"]
        det_stable = det["stable"]

        ma_x.append(x_error_px)
        ma_y.append(y_error_px)

        x_err_ma = sum(ma_x) / len(ma_x)
        y_err_ma = sum(ma_y) / len(ma_y)

        control.setXdelta(x_err_ma)
        control.setYdelta(y_err_ma)
        control.control_drone()

        centered_x = abs(x_err_ma) <= CENTER_TOL_X_PX
        centered_y = abs(y_err_ma) <= CENTER_TOL_Y_PX
        good_conf = conf >= MIN_CONFIDENCE

        good_to_drop = centered_x and centered_y and good_conf and det_stable

        if good_to_drop:
            stable_drop_counter += 1
        else:
            stable_drop_counter = max(0, stable_drop_counter - 1)

        log.info(
            "center: conf=%.3f x_err=%.2f y_err=%.2f centered_x=%s centered_y=%s stable=%s drop_counter=%d",
            conf, x_err_ma, y_err_ma,
            centered_x, centered_y, det_stable, stable_drop_counter
        )

        if stable_drop_counter >= STABLE_CONFIRM_FRAMES:
            control.stop_drone()
            return "drop_payload"

        time.sleep(0.05)

    log.info("Center timeout")
    control.stop_drone()
    return "rtl"


def drop_payload(detector):
    if control.pilot_took_over():
        return "manual_override"

    log.info("Triggering payload drop")
    detector.trigger_payload()
    time.sleep(2.0)
    return "rtl"


def rtl():
    if control.pilot_took_over():
        return "manual_override"

    log.info("Switching to RTL")
    control.stop_drone()
    control.rtl()
    time.sleep(RTL_WAIT_BEFORE_EXIT_S)
    return "done"


def manual_override():
    log.info("Manual override engaged by pilot via LOITER. Stopping autonomy.")
    try:
        control.stop_drone()
    except Exception:
        pass
    return "done"


def main():
    global STATE
    detector = None

    try:
        detector = setup()
        wait_for_arm_and_start()
        STATE = "takeoff"

        while STATE != "done":
            log.info("STATE = %s | MODE = %s", STATE, control.get_mode())

            if STATE == "takeoff":
                STATE = takeoff()

            elif STATE == "goto_target":
                STATE = goto_target()

            elif STATE == "search_target":
                STATE = search_target(detector)

            elif STATE == "center_target":
                STATE = center_target(detector)

            elif STATE == "drop_payload":
                STATE = drop_payload(detector)

            elif STATE == "rtl":
                STATE = rtl()

            elif STATE == "manual_override":
                STATE = manual_override()

            else:
                log.error("Unknown state: %s", STATE)
                break

    except KeyboardInterrupt:
        log.info("Interrupted by Ctrl+C")
        try:
            control.stop_drone()
        except Exception:
            pass

    except Exception as e:
        log.error("Mission failed: %s", str(e))
        log.error(traceback.format_exc())
        try:
            control.stop_drone()
            control.rtl()
        except Exception:
            pass

    finally:
        if detector is not None:
            detector.close()
        log.info("=== mission_mvp.py finished ===")


if __name__ == "__main__":
    main()