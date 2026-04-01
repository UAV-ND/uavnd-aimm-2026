#!/usr/bin/env python3
import sys
import time
import logging
import traceback
from collections import deque

sys.path.insert(1, '/home/uav-nano/Documents/aimm-dev/uavnd-aimm-2026')
sys.path.insert(1, '/home/uav-nano/Documents/aimm-dev/uavnd-aimm-2026/modules')

import control
from target_detector_adapter import TargetDetector

# =========================
# CONFIG
# =========================
UDP_CONNECTION = 'udpin:127.0.0.1:14552'

MISSION_WAYPOINT_MODE = "first_nav"
# "first_nav", "second_nav", "by_index"
MISSION_WAYPOINT_INDEX = 1

# This is the mission item index where AUTO should pause/loiter and hand off.
# Example:
#   0 takeoff
#   1 transit wp
#   2 loiter over target area  <- handoff
HANDOFF_MISSION_INDEX = 2

TAKEOFF_ALT = 4.0

CENTER_TOL_X_PX = 25
CENTER_TOL_Y_PX = 25
MIN_CONFIDENCE = 0.95
STABLE_CONFIRM_FRAMES = 10

CENTER_TIMEOUT_S = 60
AUTO_MONITOR_TIMEOUT_S = 300

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
TARGET_WAYPOINT = None


def setup():
    log.info("=== mission_mvp.py started ===")

    log.info("Connecting to drone...")
    control.connect_drone(UDP_CONNECTION)
    control.configure_PID("PID")
    control.set_flight_altitude(TAKEOFF_ALT)

    log.info("Starting detector...")
    detector = TargetDetector()

    return detector


def load_target_waypoint_from_fc():
    global TARGET_WAYPOINT
    log.info("Downloading mission from FC / Mission Planner upload...")
    TARGET_WAYPOINT = control.get_target_waypoint_from_mission(
        use_waypoint_mode=MISSION_WAYPOINT_MODE,
        explicit_index=MISSION_WAYPOINT_INDEX
    )
    log.info("Target waypoint loaded: %s", TARGET_WAYPOINT)


def wait_for_auto_start():
    log.info("Waiting for RC arm...")
    control.wait_until_armed()

    log.info("Reading mission upload from Mission Planner...")
    load_target_waypoint_from_fc()

    log.info("Waiting for pilot to put vehicle in AUTO and launch...")
    control.wait_for_auto_mode()


def wait_for_handoff_in_auto():
    log.info("Monitoring AUTO mission for handoff waypoint index %s", HANDOFF_MISSION_INDEX)
    start = time.time()

    while time.time() - start < AUTO_MONITOR_TIMEOUT_S:
        if control.pilot_took_over():
            return "manual_override"

        mode = control.get_mode()
        next_idx = control.get_next_mission_index()

        log.info("AUTO monitor: mode=%s next_mission_index=%s", mode, next_idx)

        if mode != "AUTO":
            log.info("Vehicle left AUTO before handoff")
            return "manual_override"

        # The FC increments commands.next as it advances.
        # At or past the handoff index means we are at/through the loiter target area.
        if next_idx is not None and int(next_idx) >= int(HANDOFF_MISSION_INDEX):
            log.info("Reached handoff waypoint area")
            return "guided_handoff"

        time.sleep(1.0)

    log.info("AUTO monitor timeout")
    return "rtl"


def guided_handoff():
    if control.pilot_took_over():
        return "manual_override"

    log.info("Switching from AUTO to GUIDED for vision phase")
    ok = control.switch_to_guided()
    if not ok:
        log.info("Failed to switch to GUIDED or pilot took over")
        return "manual_override"

    control.hold_position(1.5)
    return "center_target"


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
            log.info("No target currently visible; holding")
            control.stop_drone()
            time.sleep(0.05)
            continue

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
        wait_for_auto_start()
        STATE = "wait_for_handoff"

        while STATE != "done":
            log.info("STATE = %s | MODE = %s", STATE, control.get_mode())

            if STATE == "wait_for_handoff":
                STATE = wait_for_handoff_in_auto()

            elif STATE == "guided_handoff":
                STATE = guided_handoff()

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