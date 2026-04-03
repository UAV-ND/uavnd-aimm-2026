#!/usr/bin/env python3
# mission_mvp.py
#
# Full state flow:
#   wait_for_auto_start
#   wait_for_handoff
#   guided_handoff
#   search_for_buoy          <- 3m spiral, class=black_buoy
#   center_on_buoy           <- PID center over buoy
#   hold_at_buoy             <- 3s hold
#   search_for_payload       <- 3m spiral from buoy position, class=target
#   center_payload_target    <- PID center over target
#   drop_payload             <- GPIO trigger
#   wait_for_boat_gps        <- waits for NAMED_VALUE_FLOAT fix via MAVLink
#   goto_boat_gps            <- fly to boat, re-command if boat moves >5m
#   search_boat              <- 3m spiral, class=boat
#   align_over_boat          <- PID center over landing marker
#   descend_on_boat          <- staged lidar descent
#   final_land               <- LAND mode
#   manual_override          <- pilot LOITER takeover or any failure
#   done

import sys
import time
import math
import logging
import traceback
import threading
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
UDP_CONNECTION   = 'udpin:127.0.0.1:14552'
LIDAR_PORT       = '/dev/ttyTHS1'

PAYLOAD_TARGET_MISSION_INDEX = 2
HANDOFF_MISSION_INDEX        = 2

ENGINE_PATH = 'engine.engine'   # single model — classes: black_buoy, target, boat

# Flight altitudes
GUIDED_HOLD_ALT        = 4.0
BOAT_STAGE_ALT         = 6.0
BOAT_DESCENT_START_ALT = 5.0
BOAT_FINAL_LAND_ALT    = 1.0

MISSION_GROUNDSPEED = 1.5
BOAT_STAGE_RADIUS_M = 3.0

# ---------- Spiral (shared by all three search states) ----------
SPIRAL_STEP_M        = 3.0
SPIRAL_LEG_SPEED     = 1.0
SPIRAL_MAX_LEGS      = 16
SPIRAL_LEG_TIMEOUT_S = 30

# ---------- Buoy search / centering ----------
BUOY_SEARCH_TIMEOUT_S      = 90
BUOY_CENTER_TOL_X_PX       = 25
BUOY_CENTER_TOL_Y_PX       = 25
BUOY_MIN_CONFIDENCE        = 0.97
BUOY_STABLE_CONFIRM_FRAMES = 8
BUOY_CENTER_TIMEOUT_S      = 60
BUOY_HOLD_S                = 3.0

# ---------- Payload search / centering ----------
PAYLOAD_SEARCH_TIMEOUT_S      = 60
PAYLOAD_CENTER_TOL_X_PX       = 25
PAYLOAD_CENTER_TOL_Y_PX       = 25
PAYLOAD_MIN_CONFIDENCE        = 0.95
PAYLOAD_STABLE_CONFIRM_FRAMES = 10
PAYLOAD_CENTER_TIMEOUT_S      = 60

# ---------- Boat transit ----------
BOAT_GPS_MOVE_THRESHOLD_M = 5.0
GOTO_BOAT_TIMEOUT_S       = 120

# ---------- Boat search / align ----------
BOAT_SEARCH_TIMEOUT_S            = 90
BOAT_ALIGN_CENTER_TOL_X_PX       = 20
BOAT_ALIGN_CENTER_TOL_Y_PX       = 20
BOAT_ALIGN_MIN_CONFIDENCE        = 0.90
BOAT_ALIGN_STABLE_CONFIRM_FRAMES = 10
BOAT_ALIGN_TIMEOUT_S             = 90

# ---------- Boat descent ----------
DESCENT_STEP_M               = 0.25
DESCENT_UPDATE_PERIOD_S      = 1.0
DESCENT_CENTER_TOL_X_PX      = 20
DESCENT_CENTER_TOL_Y_PX      = 20
DESCENT_MIN_CONFIDENCE       = 0.90
DESCENT_SWITCH_TO_LAND_ALT_M = 1.2
DESCENT_TIMEOUT_S            = 180

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def offset_gps(lat, lon, north_m, east_m):
    new_lat = lat + (north_m / 111320.0)
    new_lon = lon + (east_m / (111320.0 * math.cos(math.radians(lat))))
    return new_lat, new_lon


def _run_spiral(detector, origin_lat, origin_lon, altitude, timeout_s, state_name):
    """
    Expanding square spiral centred on (origin_lat, origin_lon).
    Polls detector at ~20 Hz during transit.
    Returns ("found", det) | ("timeout", None) | ("manual_override", None).
    """
    directions          = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # N, E, S, W
    dir_idx             = 0
    leg_length          = SPIRAL_STEP_M
    legs_at_this_length = 0
    current_north       = 0.0
    current_east        = 0.0
    start               = time.time()

    for leg_num in range(SPIRAL_MAX_LEGS):
        if time.time() - start > timeout_s:
            log.info("%s: spiral timeout after %d legs", state_name, leg_num)
            return "timeout", None

        if control.pilot_took_over():
            return "manual_override", None

        dn, de       = directions[dir_idx % 4]
        target_north = current_north + dn * leg_length
        target_east  = current_east  + de * leg_length
        target_lat, target_lon = offset_gps(origin_lat, origin_lon, target_north, target_east)

        log.info(
            "%s leg %d: dir=%s dist=%.1fm  wp=(%.6f, %.6f)",
            state_name, leg_num,
            ["N", "E", "S", "W"][dir_idx % 4],
            leg_length, target_lat, target_lon
        )

        control.goto_gps_location(
            lat=target_lat, lon=target_lon, alt=altitude,
            groundspeed=SPIRAL_LEG_SPEED,
            radius_m=2.0,
            timeout_s=0
        )

        leg_start = time.time()
        while time.time() - leg_start < SPIRAL_LEG_TIMEOUT_S:
            if control.pilot_took_over():
                return "manual_override", None

            det = detector.get_target_info()
            if det is not None and det["has_target"]:
                log.info("%s: detected on leg %d  conf=%.3f",
                         state_name, leg_num, det["confidence"])
                control.stop_drone()
                return "found", det

            pos = control.get_vehicle().location.global_relative_frame
            if haversine_m(pos.lat, pos.lon, target_lat, target_lon) <= 2.0:
                break

            time.sleep(0.05)

        current_north       = target_north
        current_east        = target_east
        dir_idx            += 1
        legs_at_this_length += 1
        if legs_at_this_length == 2:
            leg_length          += SPIRAL_STEP_M
            legs_at_this_length  = 0

    log.info("%s: spiral exhausted without detection", state_name)
    control.stop_drone()
    return "timeout", None


def _center_with_detector(detector, center_tol_x, center_tol_y,
                           min_conf, stable_frames, timeout_s, state_name):
    """
    PID center loop.
    Returns ("ok", det) | ("timeout", None) | ("manual_override", None).
    """
    ma_x           = deque(maxlen=MA_X_LEN)
    ma_y           = deque(maxlen=MA_Y_LEN)
    stable_counter = 0
    start          = time.time()

    while time.time() - start < timeout_s:
        if control.pilot_took_over():
            return "manual_override", None

        det = detector.get_target_info()
        if det is None or not det["has_target"]:
            control.stop_drone()
            time.sleep(0.05)
            continue

        ma_x.append(det["x_error_px"])
        ma_y.append(det["y_error_px"])
        x_err_ma = sum(ma_x) / len(ma_x)
        y_err_ma = sum(ma_y) / len(ma_y)

        control.setXdelta(x_err_ma)
        control.setYdelta(y_err_ma)
        control.control_drone()

        centered_x = abs(x_err_ma) <= center_tol_x
        centered_y = abs(y_err_ma) <= center_tol_y
        good_conf  = det["confidence"] >= min_conf
        good       = centered_x and centered_y and good_conf and det["stable"]

        stable_counter = (stable_counter + 1) if good else max(0, stable_counter - 1)

        log.info(
            "%s: conf=%.3f xerr=%.1f yerr=%.1f cx=%s cy=%s stable=%s sc=%d",
            state_name, det["confidence"], x_err_ma, y_err_ma,
            centered_x, centered_y, det["stable"], stable_counter
        )

        if stable_counter >= stable_frames:
            control.stop_drone()
            return "ok", det

        time.sleep(0.05)

    control.stop_drone()
    return "timeout", None


# ---------------------------------------------------------------------------
# Mission states
# ---------------------------------------------------------------------------

def setup():
    log.info("=== mission_mvp.py started ===")

    log.info("Connecting to drone...")
    control.connect_drone(UDP_CONNECTION)
    control.configure_PID("PID")
    control.set_flight_altitude(GUIDED_HOLD_ALT)

    # boat_radio now listens on the DroneKit vehicle connection —
    # no serial port needed, GPS arrives via NAMED_VALUE_FLOAT over MAVLink
    log.info("Registering boat radio MAVLink listener...")
    boat_radio.connect_boat_radio(control.get_vehicle())

    log.info("Connecting lidar...")
    lidar.connect_lidar(LIDAR_PORT)

    log.info("Loading detectors...")
    buoy_detector = GenericDetector(
        engine_path=ENGINE_PATH,
        target_class="black_buoy",
        trigger_conf=BUOY_MIN_CONFIDENCE,
        enable_payload_gpio=False
    )
    payload_detector = GenericDetector(
        engine_path=ENGINE_PATH,
        target_class="target",
        trigger_conf=PAYLOAD_MIN_CONFIDENCE,
        enable_payload_gpio=True
    )
    landing_pad_detector = GenericDetector(
        engine_path=ENGINE_PATH,
        target_class="boat",
        trigger_conf=BOAT_ALIGN_MIN_CONFIDENCE,
        stable_dist_px=80,
        area_ratio_min=0.45,
        area_ratio_max=2.20,
        enable_payload_gpio=False
    )

    return buoy_detector, payload_detector, landing_pad_detector


def load_payload_waypoint_from_fc():
    global PAYLOAD_TARGET_WAYPOINT
    log.info("Downloading mission from FC...")
    PAYLOAD_TARGET_WAYPOINT = control.get_target_waypoint_from_mission(
        explicit_index=PAYLOAD_TARGET_MISSION_INDEX
    )
    log.info("Payload waypoint: %s", PAYLOAD_TARGET_WAYPOINT)


def wait_for_auto_start():
    log.info("wait_for_auto_start")
    control.wait_until_armed()
    load_payload_waypoint_from_fc()
    control.wait_for_auto_mode()
    return "wait_for_handoff"


def wait_for_handoff():
    log.info("wait_for_handoff | target index=%s", HANDOFF_MISSION_INDEX)
    start = time.time()

    while time.time() - start < AUTO_MONITOR_TIMEOUT_S:
        if control.pilot_took_over():
            return "manual_override"

        mode     = control.get_mode()
        next_idx = control.get_next_mission_index()
        log.info("AUTO: mode=%s next_idx=%s", mode, next_idx)

        if mode != "AUTO":
            log.info("Left AUTO unexpectedly")
            return "manual_override"

        if next_idx is not None and int(next_idx) >= int(HANDOFF_MISSION_INDEX):
            log.info("Handoff waypoint reached")
            return "guided_handoff"

        time.sleep(1.0)

    log.info("wait_for_handoff timeout")
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
    return "search_for_buoy"


# ---------------------------------------------------------------------------
# BUOY PHASE
# ---------------------------------------------------------------------------

def search_for_buoy(buoy_detector):
    log.info("search_for_buoy: starting spiral")
    pos       = control.get_vehicle().location.global_relative_frame
    result, _ = _run_spiral(
        detector   = buoy_detector,
        origin_lat = pos.lat,
        origin_lon = pos.lon,
        altitude   = GUIDED_HOLD_ALT,
        timeout_s  = BUOY_SEARCH_TIMEOUT_S,
        state_name = "search_for_buoy"
    )

    if result == "found":         return "center_on_buoy"
    if result == "manual_override": return "manual_override"
    log.info("search_for_buoy: buoy not found")
    return "manual_override"


def center_on_buoy(buoy_detector):
    log.info("center_on_buoy")
    result, _ = _center_with_detector(
        detector      = buoy_detector,
        center_tol_x  = BUOY_CENTER_TOL_X_PX,
        center_tol_y  = BUOY_CENTER_TOL_Y_PX,
        min_conf      = BUOY_MIN_CONFIDENCE,
        stable_frames = BUOY_STABLE_CONFIRM_FRAMES,
        timeout_s     = BUOY_CENTER_TIMEOUT_S,
        state_name    = "center_on_buoy"
    )

    if result == "manual_override": return "manual_override"
    if result == "ok":              return "hold_at_buoy"
    return "manual_override"


def hold_at_buoy():
    log.info("hold_at_buoy: holding %.1fs", BUOY_HOLD_S)
    ok = control.hold_position(BUOY_HOLD_S)
    if not ok:
        return "manual_override"
    return "search_for_payload"


# ---------------------------------------------------------------------------
# PAYLOAD PHASE
# ---------------------------------------------------------------------------

def search_for_payload(payload_detector):
    log.info("search_for_payload: starting spiral from buoy position")
    pos       = control.get_vehicle().location.global_relative_frame
    result, _ = _run_spiral(
        detector   = payload_detector,
        origin_lat = pos.lat,
        origin_lon = pos.lon,
        altitude   = GUIDED_HOLD_ALT,
        timeout_s  = PAYLOAD_SEARCH_TIMEOUT_S,
        state_name = "search_for_payload"
    )

    if result == "found":           return "center_payload_target"
    if result == "manual_override": return "manual_override"
    log.info("search_for_payload: target not found")
    return "manual_override"


def center_payload_target(payload_detector):
    log.info("center_payload_target")
    result, _ = _center_with_detector(
        detector      = payload_detector,
        center_tol_x  = PAYLOAD_CENTER_TOL_X_PX,
        center_tol_y  = PAYLOAD_CENTER_TOL_Y_PX,
        min_conf      = PAYLOAD_MIN_CONFIDENCE,
        stable_frames = PAYLOAD_STABLE_CONFIRM_FRAMES,
        timeout_s     = PAYLOAD_CENTER_TIMEOUT_S,
        state_name    = "center_payload_target"
    )

    if result == "manual_override": return "manual_override"
    if result == "ok":              return "drop_payload"
    return "manual_override"


def drop_payload(payload_detector):
    log.info("drop_payload")
    if control.pilot_took_over():
        return "manual_override"

    payload_detector.trigger_payload()
    time.sleep(2.0)
    return "wait_for_boat_gps"


# ---------------------------------------------------------------------------
# BOAT PHASE
# ---------------------------------------------------------------------------

def wait_for_boat_gps():
    """
    Block until boat_radio has a valid GPS fix from the MAVLink stream.
    The fix may already be present if the boat was transmitting before drop.
    """
    log.info("wait_for_boat_gps")
    start = time.time()

    while time.time() - start < 30:
        if control.pilot_took_over():
            return "manual_override"

        if boat_radio.read_boat_gps() is not None:
            log.info("Boat GPS ready: %s", boat_radio.read_boat_gps())
            return "goto_boat_gps"

        time.sleep(0.1)

    log.info("wait_for_boat_gps: timeout — no fix received")
    return "manual_override"


def goto_boat_gps():
    """
    Fly toward the boat. Re-commands whenever boat moves >BOAT_GPS_MOVE_THRESHOLD_M.
    """
    log.info("goto_boat_gps")
    if control.pilot_took_over():
        return "manual_override"

    control.set_flight_altitude(BOAT_STAGE_ALT)

    gps = boat_radio.read_boat_gps()
    if gps is None:
        return "wait_for_boat_gps"

    cmd_lat, cmd_lon = gps["lat"], gps["lon"]
    control.goto_gps_location(
        lat=cmd_lat, lon=cmd_lon, alt=BOAT_STAGE_ALT,
        groundspeed=MISSION_GROUNDSPEED,
        radius_m=999.0, timeout_s=1
    )
    log.info("Initial goto: (%.6f, %.6f)", cmd_lat, cmd_lon)

    start = time.time()
    while time.time() - start < GOTO_BOAT_TIMEOUT_S:
        if control.pilot_took_over():
            return "manual_override"

        gps = boat_radio.read_boat_gps()
        if gps is not None:
            moved = haversine_m(cmd_lat, cmd_lon, gps["lat"], gps["lon"])
            if moved > BOAT_GPS_MOVE_THRESHOLD_M:
                cmd_lat, cmd_lon = gps["lat"], gps["lon"]
                log.info("Boat moved %.1fm — re-commanding (%.6f, %.6f)",
                         moved, cmd_lat, cmd_lon)
                control.goto_gps_location(
                    lat=cmd_lat, lon=cmd_lon, alt=BOAT_STAGE_ALT,
                    groundspeed=MISSION_GROUNDSPEED,
                    radius_m=999.0, timeout_s=1
                )

        pos    = control.get_vehicle().location.global_relative_frame
        dist_m = haversine_m(pos.lat, pos.lon, cmd_lat, cmd_lon)
        log.info("Distance to boat: %.1fm", dist_m)

        if dist_m <= BOAT_STAGE_RADIUS_M:
            log.info("Arrived at boat staging position")
            control.hold_position(2.0)
            return "search_boat"

        time.sleep(1.0)

    log.info("goto_boat_gps: timeout")
    return "manual_override"


def search_boat(landing_pad_detector):
    log.info("search_boat: starting spiral")
    pos       = control.get_vehicle().location.global_relative_frame
    result, _ = _run_spiral(
        detector   = landing_pad_detector,
        origin_lat = pos.lat,
        origin_lon = pos.lon,
        altitude   = BOAT_STAGE_ALT,
        timeout_s  = BOAT_SEARCH_TIMEOUT_S,
        state_name = "search_boat"
    )

    if result == "found":           return "align_over_boat"
    if result == "manual_override": return "manual_override"
    log.info("search_boat: landing marker not found")
    return "manual_override"


def align_over_boat(landing_pad_detector):
    log.info("align_over_boat")
    control.set_flight_altitude(BOAT_DESCENT_START_ALT)

    result, _ = _center_with_detector(
        detector      = landing_pad_detector,
        center_tol_x  = BOAT_ALIGN_CENTER_TOL_X_PX,
        center_tol_y  = BOAT_ALIGN_CENTER_TOL_Y_PX,
        min_conf      = BOAT_ALIGN_MIN_CONFIDENCE,
        stable_frames = BOAT_ALIGN_STABLE_CONFIRM_FRAMES,
        timeout_s     = BOAT_ALIGN_TIMEOUT_S,
        state_name    = "align_over_boat"
    )

    if result == "manual_override": return "manual_override"
    if result == "ok":              return "descend_on_boat"
    return "manual_override"


def descend_on_boat(landing_pad_detector):
    log.info("descend_on_boat")
    start             = time.time()
    last_descent_time = 0.0
    ma_x              = deque(maxlen=MA_X_LEN)
    ma_y              = deque(maxlen=MA_Y_LEN)

    while time.time() - start < DESCENT_TIMEOUT_S:
        if control.pilot_took_over():
            return "manual_override"

        det = landing_pad_detector.get_target_info()
        if det is None or not det["has_target"]:
            log.info("descend_on_boat: lost landing marker — holding")
            control.stop_drone()
            time.sleep(0.1)
            continue

        lidar_dist, _ = lidar.read_lidar_distance()
        if lidar_dist is None:
            log.warning("descend_on_boat: lidar read failed — holding")
            control.stop_drone()
            time.sleep(0.1)
            continue

        ma_x.append(det["x_error_px"])
        ma_y.append(det["y_error_px"])
        x_err_ma = sum(ma_x) / len(ma_x)
        y_err_ma = sum(ma_y) / len(ma_y)

        control.setXdelta(x_err_ma)
        control.setYdelta(y_err_ma)
        control.control_drone()

        centered_x = abs(x_err_ma) <= DESCENT_CENTER_TOL_X_PX
        centered_y = abs(y_err_ma) <= DESCENT_CENTER_TOL_Y_PX
        good       = (centered_x and centered_y
                      and det["confidence"] >= DESCENT_MIN_CONFIDENCE
                      and det["stable"])

        log.info(
            "descend_on_boat: conf=%.3f xerr=%.1f yerr=%.1f lidar=%.2f "
            "cx=%s cy=%s stable=%s alt=%.2f",
            det["confidence"], x_err_ma, y_err_ma, lidar_dist,
            centered_x, centered_y, det["stable"], control.get_flight_altitude()
        )

        if good and (time.time() - last_descent_time) >= DESCENT_UPDATE_PERIOD_S:
            new_alt = max(BOAT_FINAL_LAND_ALT, control.get_flight_altitude() - DESCENT_STEP_M)
            control.set_flight_altitude(new_alt)
            last_descent_time = time.time()
            log.info("Descent step -> alt=%.2f", new_alt)

        if lidar_dist <= DESCENT_SWITCH_TO_LAND_ALT_M and good:
            control.stop_drone()
            return "final_land"

        time.sleep(0.05)

    log.info("descend_on_boat: timeout")
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
    log.info("manual_override: stopping drone")
    try:
        control.stop_drone()
    except Exception:
        pass
    return "done"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global STATE
    buoy_detector        = None
    payload_detector     = None
    landing_pad_detector = None

    try:
        buoy_detector, payload_detector, landing_pad_detector = setup()
        STATE = "wait_for_auto_start"

        while STATE != "done":
            log.info("── STATE: %-30s MODE: %s", STATE, control.get_mode())

            if   STATE == "wait_for_auto_start":  STATE = wait_for_auto_start()
            elif STATE == "wait_for_handoff":      STATE = wait_for_handoff()
            elif STATE == "guided_handoff":        STATE = guided_handoff()

            elif STATE == "search_for_buoy":       STATE = search_for_buoy(buoy_detector)
            elif STATE == "center_on_buoy":        STATE = center_on_buoy(buoy_detector)
            elif STATE == "hold_at_buoy":          STATE = hold_at_buoy()

            elif STATE == "search_for_payload":    STATE = search_for_payload(payload_detector)
            elif STATE == "center_payload_target": STATE = center_payload_target(payload_detector)
            elif STATE == "drop_payload":          STATE = drop_payload(payload_detector)

            elif STATE == "wait_for_boat_gps":     STATE = wait_for_boat_gps()
            elif STATE == "goto_boat_gps":         STATE = goto_boat_gps()
            elif STATE == "search_boat":           STATE = search_boat(landing_pad_detector)
            elif STATE == "align_over_boat":       STATE = align_over_boat(landing_pad_detector)
            elif STATE == "descend_on_boat":       STATE = descend_on_boat(landing_pad_detector)
            elif STATE == "final_land":            STATE = final_land()
            elif STATE == "manual_override":       STATE = manual_override()

            else:
                log.error("Unknown state: %s", STATE)
                break

    except KeyboardInterrupt:
        log.info("Ctrl+C — stopping")
        try:
            control.stop_drone()
        except Exception:
            pass

    except Exception as e:
        log.error("Mission failed: %s", e)
        log.error(traceback.format_exc())
        try:
            control.stop_drone()
        except Exception:
            pass

    finally:
        try:
            boat_radio.disconnect_boat_radio()
        except Exception:
            pass

        for det in [buoy_detector, payload_detector, landing_pad_detector]:
            try:
                if det is not None:
                    det.close()
            except Exception:
                pass

        try:
            lidar.disconnect_lidar()
        except Exception:
            pass

        log.info("=== mission_mvp.py finished ===")


if __name__ == "__main__":
    main()