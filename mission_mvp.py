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
import boat_radio
from detector_adapter import GenericDetector

# =========================
# CONFIG
# =========================
UDP_CONNECTION = 'udpin:127.0.0.1:14552'
LIDAR_PORT = '/dev/ttyTHS1'
BOAT_RADIO_PORT = '/dev/ttyUSB0'
BOAT_RADIO_BAUD = 57600

# Mission Planner AUTO mission structure
# Example:
# 0 takeoff
# 1 transit
# 2 loiter over payload area <-- handoff point
PAYLOAD_TARGET_MISSION_INDEX = 2
HANDOFF_MISSION_INDEX = 2

# Engines
PAYLOAD_ENGINE_PATH = 'engine.engine'
LANDING_PAD_ENGINE_PATH = 'landing_pad.engine'

# Flight altitudes
GUIDED_HOLD_ALT = 4.0
BOAT_STAGE_ALT = 6.0
BOAT_DESCENT_START_ALT = 5.0
BOAT_FINAL_LAND_ALT = 1.0

MISSION_GROUNDSPEED = 1.5
BOAT_STAGE_RADIUS_M = 3.0

# Payload target centering
PAYLOAD_CENTER_TOL_X_PX = 25
PAYLOAD_CENTER_TOL_Y_PX = 25
PAYLOAD_MIN_CONFIDENCE = 0.95
PAYLOAD_STABLE_CONFIRM_FRAMES = 10
PAYLOAD_CENTER_TIMEOUT_S = 60

# Boat search / align
BOAT_SEARCH_TIMEOUT_S = 90
BOAT_ALIGN_CENTER_TOL_X_PX = 20
BOAT_ALIGN_CENTER_TOL_Y_PX = 20
BOAT_ALIGN_MIN_CONFIDENCE = 0.90
BOAT_ALIGN_STABLE_CONFIRM_FRAMES = 10
BOAT_ALIGN_TIMEOUT_S = 90

# Boat descent
DESCENT_STEP_M = 0.25
DESCENT_UPDATE_PERIOD_S = 1.0
DESCENT_CENTER_TOL_X_PX = 20
DESCENT_CENTER_TOL_Y_PX = 20
DESCENT_MIN_CONFIDENCE = 0.90
DESCENT_SWITCH_TO_LAND_ALT_M = 1.2
DESCENT_TIMEOUT_S = 180

# Moving average windows
MA_X_LEN = 5
MA_Y_LEN = 5

AUTO_MONITOR_TIMEOUT_S = 300

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
PAYLOAD_TARGET_WAYPOINT = None
LATEST_BOAT_GPS = None


def setup():
    log.info("=== mission_mvp.py started ===")

    log.info("Connecting to drone...")
    control.connect_drone(UDP_CONNECTION)
    control.configure_PID("PID")
    control.set_flight_altitude(GUIDED_HOLD_ALT)

    log.info("Connecting lidar...")
    lidar.connect_lidar(LIDAR_PORT)

    log.info("Connecting boat radio...")
    boat_radio.connect_boat_radio(BOAT_RADIO_PORT, baudrate=BOAT_RADIO_BAUD, timeout=1.0)

    log.info("Starting payload detector...")
    payload_detector = GenericDetector(
        engine_path=PAYLOAD_ENGINE_PATH,
        enable_payload_gpio=True
    )

    log.info("Starting landing pad detector...")
    landing_pad_detector = GenericDetector(
        engine_path=LANDING_PAD_ENGINE_PATH,
        trigger_conf=0.90,
        stable_dist_px=80,
        area_ratio_min=0.45,
        area_ratio_max=2.20,
        enable_payload_gpio=False
    )

    return payload_detector, landing_pad_detector


def load_payload_waypoint_from_fc():
    global PAYLOAD_TARGET_WAYPOINT
    log.info("Downloading mission from FC / Mission Planner upload...")
    PAYLOAD_TARGET_WAYPOINT = control.get_target_waypoint_from_mission(
        explicit_index=PAYLOAD_TARGET_MISSION_INDEX
    )
    log.info("Payload target waypoint loaded: %s", PAYLOAD_TARGET_WAYPOINT)


def wait_for_auto_start():
    log.info("wait_for_auto_start")
    control.wait_until_armed()
    load_payload_waypoint_from_fc()
    control.wait_for_auto_mode()
    return "wait_for_handoff"


def wait_for_handoff():
    log.info("wait_for_handoff | monitoring AUTO for handoff index %s", HANDOFF_MISSION_INDEX)
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

        if next_idx is not None and int(next_idx) >= int(HANDOFF_MISSION_INDEX):
            log.info("Reached handoff waypoint area")
            return "guided_handoff"

        time.sleep(1.0)

    log.info("AUTO monitor timeout")
    return "manual_override"


def guided_handoff():
    log.info("guided_handoff")
    if control.pilot_took_over():
        return "manual_override"

    ok = control.switch_to_guided()
    if not ok:
        return "manual_override"

    control.set_flight_altitude(GUIDED_HOLD_ALT)
    control.hold_position(1.5)
    return "center_payload_target"


def _center_with_detector(detector, center_tol_x, center_tol_y, min_conf, stable_frames, timeout_s, state_name):
    ma_x = deque(maxlen=MA_X_LEN)
    ma_y = deque(maxlen=MA_Y_LEN)
    stable_counter = 0
    start = time.time()

    while time.time() - start < timeout_s:
        if control.pilot_took_over():
            return "manual_override", None

        det = detector.get_target_info()

        if det is None or not det["has_target"]:
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

        centered_x = abs(x_err_ma) <= center_tol_x
        centered_y = abs(y_err_ma) <= center_tol_y
        good_conf = conf >= min_conf
        good = centered_x and centered_y and good_conf and det_stable

        if good:
            stable_counter += 1
        else:
            stable_counter = max(0, stable_counter - 1)

        log.info(
            "%s: conf=%.3f x_err=%.2f y_err=%.2f centered_x=%s centered_y=%s stable=%s stable_counter=%d",
            state_name, conf, x_err_ma, y_err_ma,
            centered_x, centered_y, det_stable, stable_counter
        )

        if stable_counter >= stable_frames:
            control.stop_drone()
            return "ok", det

        time.sleep(0.05)

    control.stop_drone()
    return "timeout", None


def center_payload_target(payload_detector):
    log.info("center_payload_target")
    result, _ = _center_with_detector(
        detector=payload_detector,
        center_tol_x=PAYLOAD_CENTER_TOL_X_PX,
        center_tol_y=PAYLOAD_CENTER_TOL_Y_PX,
        min_conf=PAYLOAD_MIN_CONFIDENCE,
        stable_frames=PAYLOAD_STABLE_CONFIRM_FRAMES,
        timeout_s=PAYLOAD_CENTER_TIMEOUT_S,
        state_name="payload_center"
    )

    if result == "manual_override":
        return "manual_override"
    if result == "ok":
        return "drop_payload"
    return "manual_override"


def drop_payload(payload_detector):
    log.info("drop_payload")
    if control.pilot_took_over():
        return "manual_override"

    payload_detector.trigger_payload()
    time.sleep(2.0)
    return "wait_for_boat_gps"


def wait_for_boat_gps():
    global LATEST_BOAT_GPS
    log.info("wait_for_boat_gps")
    start = time.time()

    while time.time() - start < 30:
        if control.pilot_took_over():
            return "manual_override"

        msg = boat_radio.read_boat_gps()
        if msg is not None:
            LATEST_BOAT_GPS = msg
            log.info("Boat GPS received: %s", LATEST_BOAT_GPS)
            return "goto_boat_gps"

        time.sleep(0.1)

    log.info("No boat GPS received")
    return "manual_override"


def goto_boat_gps():
    global LATEST_BOAT_GPS
    log.info("goto_boat_gps")

    if control.pilot_took_over():
        return "manual_override"

    if LATEST_BOAT_GPS is None:
        return "wait_for_boat_gps"

    control.set_flight_altitude(BOAT_STAGE_ALT)

    ok = control.goto_gps_location(
        lat=LATEST_BOAT_GPS["lat"],
        lon=LATEST_BOAT_GPS["lon"],
        alt=BOAT_STAGE_ALT,
        groundspeed=MISSION_GROUNDSPEED,
        radius_m=BOAT_STAGE_RADIUS_M,
        timeout_s=120
    )

    if not ok:
        return "manual_override"

    control.hold_position(2.0)
    return "search_boat"


def search_boat(landing_pad_detector):
    log.info("search_boat")
    start = time.time()

    while time.time() - start < BOAT_SEARCH_TIMEOUT_S:
        if control.pilot_took_over():
            return "manual_override"

        det = landing_pad_detector.get_target_info()
        if det is not None and det["has_target"]:
            log.info("Landing pad detected: conf=%.3f stable_counter=%d",
                     det["confidence"], det["stable_counter"])
            return "align_over_boat"

        control.stop_drone()
        time.sleep(0.05)

    log.info("Boat search timeout")
    return "manual_override"


def align_over_boat(landing_pad_detector):
    log.info("align_over_boat")
    control.set_flight_altitude(BOAT_DESCENT_START_ALT)

    result, _ = _center_with_detector(
        detector=landing_pad_detector,
        center_tol_x=BOAT_ALIGN_CENTER_TOL_X_PX,
        center_tol_y=BOAT_ALIGN_CENTER_TOL_Y_PX,
        min_conf=BOAT_ALIGN_MIN_CONFIDENCE,
        stable_frames=BOAT_ALIGN_STABLE_CONFIRM_FRAMES,
        timeout_s=BOAT_ALIGN_TIMEOUT_S,
        state_name="boat_align"
    )

    if result == "manual_override":
        return "manual_override"
    if result == "ok":
        return "descend_on_boat"
    return "manual_override"


def descend_on_boat(landing_pad_detector):
    log.info("descend_on_boat")
    start = time.time()
    last_descent_time = 0.0

    ma_x = deque(maxlen=MA_X_LEN)
    ma_y = deque(maxlen=MA_Y_LEN)

    while time.time() - start < DESCENT_TIMEOUT_S:
        if control.pilot_took_over():
            return "manual_override"

        det = landing_pad_detector.get_target_info()
        if det is None or not det["has_target"]:
            log.info("Lost landing pad during descent, holding")
            control.stop_drone()
            time.sleep(0.1)
            continue

        x_error_px = det["x_error_px"]
        y_error_px = det["y_error_px"]
        conf = det["confidence"]
        det_stable = det["stable"]

        lidar_dist, _ = lidar.read_lidar_distance()

        ma_x.append(x_error_px)
        ma_y.append(y_error_px)

        x_err_ma = sum(ma_x) / len(ma_x)
        y_err_ma = sum(ma_y) / len(ma_y)

        control.setXdelta(x_err_ma)
        control.setYdelta(y_err_ma)
        control.control_drone()

        centered_x = abs(x_err_ma) <= DESCENT_CENTER_TOL_X_PX
        centered_y = abs(y_err_ma) <= DESCENT_CENTER_TOL_Y_PX
        good_conf = conf >= DESCENT_MIN_CONFIDENCE
        good = centered_x and centered_y and good_conf and det_stable

        log.info(
            "boat_descent: conf=%.3f x_err=%.2f y_err=%.2f lidar=%.2f centered_x=%s centered_y=%s stable=%s cmd_alt=%.2f",
            conf, x_err_ma, y_err_ma, lidar_dist,
            centered_x, centered_y, det_stable, control.get_flight_altitude()
        )

        if good and (time.time() - last_descent_time) >= DESCENT_UPDATE_PERIOD_S:
            new_alt = max(BOAT_FINAL_LAND_ALT, control.get_flight_altitude() - DESCENT_STEP_M)
            control.set_flight_altitude(new_alt)
            last_descent_time = time.time()
            log.info("Descent step -> new commanded altitude %.2f", new_alt)

        if lidar_dist <= DESCENT_SWITCH_TO_LAND_ALT_M and good:
            control.stop_drone()
            return "final_land"

        time.sleep(0.05)

    log.info("Descent timeout")
    control.stop_drone()
    return "manual_override"


def final_land():
    log.info("final_land")
    if control.pilot_took_over():
        return "manual_override"

    ok = control.final_land()
    if not ok:
        return "manual_override"

    time.sleep(10.0)
    return "done"


def manual_override():
    log.info("manual_override")
    try:
        control.stop_drone()
    except Exception:
        pass
    return "done"


def main():
    global STATE
    payload_detector = None
    landing_pad_detector = None

    try:
        payload_detector, landing_pad_detector = setup()
        STATE = "wait_for_auto_start"

        while STATE != "done":
            log.info("STATE = %s | MODE = %s", STATE, control.get_mode())

            if STATE == "wait_for_auto_start":
                STATE = wait_for_auto_start()

            elif STATE == "wait_for_handoff":
                STATE = wait_for_handoff()

            elif STATE == "guided_handoff":
                STATE = guided_handoff()

            elif STATE == "center_payload_target":
                STATE = center_payload_target(payload_detector)

            elif STATE == "drop_payload":
                STATE = drop_payload(payload_detector)

            elif STATE == "wait_for_boat_gps":
                STATE = wait_for_boat_gps()

            elif STATE == "goto_boat_gps":
                STATE = goto_boat_gps()

            elif STATE == "search_boat":
                STATE = search_boat(landing_pad_detector)

            elif STATE == "align_over_boat":
                STATE = align_over_boat(landing_pad_detector)

            elif STATE == "descend_on_boat":
                STATE = descend_on_boat(landing_pad_detector)

            elif STATE == "final_land":
                STATE = final_land()

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
        except Exception:
            pass

    finally:
        try:
            if payload_detector is not None:
                payload_detector.close()
        except Exception:
            pass

        try:
            if landing_pad_detector is not None:
                landing_pad_detector.close()
        except Exception:
            pass

        try:
            boat_radio.disconnect_boat_radio()
        except Exception:
            pass

        try:
            lidar.disconnect_lidar()
        except Exception:
            pass

        log.info("=== mission_mvp.py finished ===")


if __name__ == "__main__":
    main()