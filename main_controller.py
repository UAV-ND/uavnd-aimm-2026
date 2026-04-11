#!/usr/bin/env python3
# main_controller.py
#
# Main controller for the UAVND AIMM mission.
#
# States:
#   wait_for_auto_start
#   wait_for_handoff
#   guided_handoff
#   search_for_buoy
#   center_on_buoy
#   hold_at_buoy
#   search_for_payload
#   center_payload_target
#   drop_payload
#   return_to_launch          <- RTL mode (home), then wait for disarm
#   manual_override
#   done
#
# Bench checklist (Messages tab + main_controller.log, no traceback):
#   1) AIMM: link OK -> AIMM: detectors ready
#   2) After arm: AIMM: armed -> AIMM: dl mission -> Mission snapshot lines
#      -> AIMM: n{count} cmd0/cmd1/cmd2 -> AIMM: WP ... -> AIMM: AUTO ready
#      -> AIMM: RC drop armed
#   3) In AUTO: AIMM: handoff n>=... -> AIMM: GUIDED hold -> search/center states
#
# Mission rows vs PAYLOAD_TARGET_MISSION_INDEX / HANDOFF_MISSION_INDEX:
#   Use 0-based DroneKit order after download. MP "row 1" TAKEOFF + "row 2" WP
#   only => indices 0 and 1 (waypoint = 1). If HOME is stored as item 0 in the
#   downloaded list, waypoint shifts to index 2 — match constants to snapshot.


import sys
import time
import math
import logging
import traceback
from collections import deque

sys.path.insert(1, '/home/uav-nano/Documents/aimm-dev/uavnd-aimm-2026')
sys.path.insert(1, '/home/uav-nano/Documents/aimm-dev/uavnd-aimm-2026/modules')
_repo = '/home/uav-nano/Documents/aimm-dev/uavnd-aimm-2026'
sys.path.insert(1, _repo)
sys.path.insert(1, _repo + '/modules')
sys.path.insert(1, _repo + '/cv/models')

from pymavlink import mavutil

import control
import drone
from detector_adapter import GenericDetector


def _gcs(msg, severity=None):
    """STATUSTEXT to GCS; keep msg short (total truncated to 50 chars in drone layer)."""
    drone.send_statustext(msg, severity=severity)

# =========================
# CONFIG
# =========================
UDP_CONNECTION = 'udpin:127.0.0.1:14552'

# 0-based DroneKit mission index of NAV_WAYPOINT (e.g. TAKEOFF=0, WP=1).
PAYLOAD_TARGET_MISSION_INDEX = 1
HANDOFF_MISSION_INDEX        = 1

ENGINE_PATH = 'engine.engine'

GUIDED_HOLD_ALT = 4.0

SPIRAL_STEP_M        = 2.0
SPIRAL_LEG_SPEED     = 1.0
SPIRAL_MAX_LEGS      = 16
SPIRAL_LEG_TIMEOUT_S = 30

BUOY_SEARCH_TIMEOUT_S      = 90
BUOY_CENTER_TOL_X_PX       = 25
BUOY_CENTER_TOL_Y_PX       = 25
BUOY_MIN_CONFIDENCE        = 0.97
BUOY_STABLE_CONFIRM_FRAMES = 8
BUOY_CENTER_TIMEOUT_S      = 60
BUOY_HOLD_S                = 3.0

PAYLOAD_SEARCH_TIMEOUT_S      = 60
PAYLOAD_CENTER_TOL_X_PX       = 25
PAYLOAD_CENTER_TOL_Y_PX       = 25
PAYLOAD_MIN_CONFIDENCE        = 0.95
PAYLOAD_STABLE_CONFIRM_FRAMES = 10
PAYLOAD_CENTER_TIMEOUT_S      = 60

MA_X_LEN = 5
MA_Y_LEN = 5

AUTO_MONITOR_TIMEOUT_S = 300
RTL_WAIT_TIMEOUT_S     = 600

log_path = '/home/uav-nano/Documents/aimm-dev/uavnd-aimm-2026/main_controller.log'
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
    directions          = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx             = 0
    leg_length          = SPIRAL_STEP_M
    legs_at_this_length = 0
    current_north       = 0.0
    current_east        = 0.0
    start               = time.time()

    for leg_num in range(SPIRAL_MAX_LEGS):
        if time.time() - start > timeout_s:
            log.info("%s: spiral timeout after %d legs", state_name, leg_num)
            _gcs(
                "AIMM: {} spiral tmo".format(state_name[:22]),
                mavutil.mavlink.MAV_SEVERITY_WARNING,
            )
            return "timeout", None

        if control.rc_manual_drop_requested():
            control.stop_drone()
            return "manual_drop", None
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
            if control.rc_manual_drop_requested():
                control.stop_drone()
                return "manual_drop", None
            if control.pilot_took_over():
                return "manual_override", None

            det = detector.get_target_info()
            if det is not None and det["has_target"]:
                log.info("%s: detected on leg %d  conf=%.3f",
                         state_name, leg_num, det["confidence"])
                if state_name == "search_for_buoy":
                    _gcs("AIMM: buoy detected", mavutil.mavlink.MAV_SEVERITY_NOTICE)
                elif state_name == "search_for_payload":
                    _gcs("AIMM: payload seen", mavutil.mavlink.MAV_SEVERITY_NOTICE)
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
    _gcs(
        "AIMM: {} no detect".format(state_name[:20]),
        mavutil.mavlink.MAV_SEVERITY_WARNING,
    )
    control.stop_drone()
    return "timeout", None


def _center_with_detector(detector, center_tol_x, center_tol_y,
                          min_conf, stable_frames, timeout_s, state_name):
    ma_x           = deque(maxlen=MA_X_LEN)
    ma_y           = deque(maxlen=MA_Y_LEN)
    stable_counter = 0
    start          = time.time()

    while time.time() - start < timeout_s:
        if control.rc_manual_drop_requested():
            control.stop_drone()
            return "manual_drop", None
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
            if state_name == "center_on_buoy":
                _gcs("AIMM: buoy locked", mavutil.mavlink.MAV_SEVERITY_NOTICE)
            elif state_name == "center_payload_target":
                _gcs("AIMM: payload locked", mavutil.mavlink.MAV_SEVERITY_NOTICE)
            control.stop_drone()
            return "ok", det

        time.sleep(0.05)

    control.stop_drone()
    _gcs(
        "AIMM: {} center tmo".format(state_name[:20]),
        mavutil.mavlink.MAV_SEVERITY_WARNING,
    )
    return "timeout", None


# ---------------------------------------------------------------------------
# Mission states
# ---------------------------------------------------------------------------

def setup():
    log.info("=== main_controller.py started ===")

    log.info("Connecting to drone...")
    control.connect_drone(UDP_CONNECTION)
    _gcs("AIMM: link OK", mavutil.mavlink.MAV_SEVERITY_NOTICE)
    control.configure_PID("PID")
    control.set_flight_altitude(GUIDED_HOLD_ALT)

    log.info("Loading detectors (buoy + payload only)...")
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

    _gcs("AIMM: detectors ready", mavutil.mavlink.MAV_SEVERITY_INFO)
    return buoy_detector, payload_detector


def load_payload_waypoint_from_fc():
    global PAYLOAD_TARGET_WAYPOINT
    log.info("Downloading mission from FC...")
    _gcs("AIMM: dl mission", mavutil.mavlink.MAV_SEVERITY_INFO)
    mission_items = drone.list_mission_commands()
    drone.emit_mission_head_snapshot(mission_items, log.info, _gcs)
    PAYLOAD_TARGET_WAYPOINT = control.get_target_waypoint_from_mission(
        explicit_index=PAYLOAD_TARGET_MISSION_INDEX,
        mission_items=mission_items,
    )
    log.info("Payload waypoint: %s", PAYLOAD_TARGET_WAYPOINT)
    wp = PAYLOAD_TARGET_WAYPOINT
    try:
        lat = float(wp["lat"])
        lon = float(wp["lon"])
        alt = float(wp["alt"])
        idx = int(wp.get("seq", PAYLOAD_TARGET_MISSION_INDEX))
        # STATUSTEXT payload max 50 chars
        line = "AIMM: WP i{} a{:.0f} {:.4f},{:.4f}".format(idx, alt, lat, lon)
        _gcs(line[:50], mavutil.mavlink.MAV_SEVERITY_NOTICE)
    except (TypeError, ValueError, KeyError):
        _gcs("AIMM: payload WP loaded", mavutil.mavlink.MAV_SEVERITY_NOTICE)


def wait_for_auto_start():
    log.info("wait_for_auto_start")
    _gcs("AIMM: waiting for auto start", mavutil.mavlink.MAV_SEVERITY_INFO)
    control.wait_until_armed()
    _gcs("AIMM: armed", mavutil.mavlink.MAV_SEVERITY_NOTICE)
    load_payload_waypoint_from_fc()
    control.wait_for_auto_mode()
    _gcs("AIMM: AUTO ready", mavutil.mavlink.MAV_SEVERITY_NOTICE)
    control.enable_manual_drop_via_rc(True)
    _gcs("AIMM: RC drop armed", mavutil.mavlink.MAV_SEVERITY_INFO)
    return "wait_for_handoff"


def wait_for_handoff():
    log.info("wait_for_handoff | target index=%s", HANDOFF_MISSION_INDEX)
    start = time.time()

    while time.time() - start < AUTO_MONITOR_TIMEOUT_S:
        if control.rc_manual_drop_requested():
            return "drop_payload"
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
            _gcs(
                "AIMM: handoff n>={}".format(int(HANDOFF_MISSION_INDEX))[:50],
                mavutil.mavlink.MAV_SEVERITY_NOTICE,
            )
            return "guided_handoff"

        time.sleep(1.0)

    log.info("wait_for_handoff timeout")
    _gcs("AIMM: handoff wait tmo", mavutil.mavlink.MAV_SEVERITY_WARNING)
    return "manual_override"


def guided_handoff():
    log.info("guided_handoff")
    if control.pilot_took_over():
        _gcs("AIMM: takeover b4 GUIDED", mavutil.mavlink.MAV_SEVERITY_WARNING)
        return "manual_override"

    ok = control.switch_to_guided()
    if not ok:
        _gcs("AIMM: GUIDED failed", mavutil.mavlink.MAV_SEVERITY_ERROR)
        return "manual_override"

    _gcs("AIMM: GUIDED hold", mavutil.mavlink.MAV_SEVERITY_NOTICE)
    control.set_flight_altitude(GUIDED_HOLD_ALT)
    hp = control.hold_position(1.5)
    if hp == "manual_drop":
        return "drop_payload"
    if not hp:
        _gcs("AIMM: hold abort", mavutil.mavlink.MAV_SEVERITY_WARNING)
        return "manual_override"
    return "search_for_buoy"


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

    if result == "found":
        return "center_on_buoy"
    if result == "manual_drop":
        return "drop_payload"
    if result == "manual_override":
        return "manual_override"
    log.info("search_for_buoy: buoy not found")
    _gcs("AIMM: no buoy", mavutil.mavlink.MAV_SEVERITY_WARNING)
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

    if result == "manual_drop":
        return "drop_payload"
    if result == "manual_override":
        return "manual_override"
    if result == "ok":
        return "hold_at_buoy"
    return "manual_override"


def hold_at_buoy():
    log.info("hold_at_buoy: holding %.1fs", BUOY_HOLD_S)
    _gcs("AIMM: hold at buoy", mavutil.mavlink.MAV_SEVERITY_INFO)
    ok = control.hold_position(BUOY_HOLD_S)
    if ok == "manual_drop":
        return "drop_payload"
    if not ok:
        return "manual_override"
    return "search_for_payload"


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

    if result == "found":
        return "center_payload_target"
    if result == "manual_drop":
        return "drop_payload"
    if result == "manual_override":
        return "manual_override"
    log.info("search_for_payload: target not found")
    _gcs("AIMM: no payload tgt", mavutil.mavlink.MAV_SEVERITY_WARNING)
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

    if result == "manual_drop":
        return "drop_payload"
    if result == "manual_override":
        return "manual_override"
    if result == "ok":
        return "drop_payload"
    return "manual_override"


def drop_payload(payload_detector):
    log.info("drop_payload")
    _gcs("AIMM: DROP", mavutil.mavlink.MAV_SEVERITY_NOTICE)
    payload_detector.trigger_payload()
    time.sleep(2.0)
    return "return_to_launch"


def return_to_launch():
    """
    Command RTL; block until disarmed (landing complete) or timeout.
    """
    log.info("return_to_launch")
    _gcs("AIMM: RTL", mavutil.mavlink.MAV_SEVERITY_NOTICE)
    if control.rc_manual_drop_requested():
        return "drop_payload"
    if control.pilot_took_over():
        return "manual_override"

    control.stop_drone()
    ok = control.return_to_launch()
    if ok == "manual_drop":
        return "drop_payload"
    if not ok:
        _gcs("AIMM: RTL cmd failed", mavutil.mavlink.MAV_SEVERITY_WARNING)
        return "manual_override"

    start = time.time()
    while time.time() - start < RTL_WAIT_TIMEOUT_S:
        if control.rc_manual_drop_requested():
            return "drop_payload"
        if control.pilot_took_over():
            return "manual_override"

        vehicle = control.get_vehicle()
        if not vehicle.armed:
            log.info("return_to_launch: disarmed — mission complete")
            _gcs("AIMM: landed disarm", mavutil.mavlink.MAV_SEVERITY_NOTICE)
            return "done"

        log.info("return_to_launch: MODE=%s armed=%s", control.get_mode(), vehicle.armed)
        time.sleep(1.0)

    log.warning("return_to_launch: timeout after %ds (still armed)", RTL_WAIT_TIMEOUT_S)
    _gcs("AIMM: RTL wait tmo", mavutil.mavlink.MAV_SEVERITY_WARNING)
    return "done"


def manual_override(_payload_detector):
    log.info(
        "manual_override: stopping drone; waiting for RC ch%s payload command (no timeout)",
        control.MANUAL_DROP_RC_CHANNEL,
    )
    _gcs("AIMM: manual override", mavutil.mavlink.MAV_SEVERITY_WARNING)
    try:
        control.stop_drone()
    except Exception:
        pass
    while True:
        if control.rc_manual_drop_requested():
            return "drop_payload"
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global STATE
    buoy_detector    = None
    payload_detector = None
    prev_state       = None

    try:
        buoy_detector, payload_detector = setup()
        STATE = "wait_for_auto_start"

        while STATE != "done":
            if STATE != prev_state:
                _gcs(
                    "AIMM: {}".format(STATE)[:50],
                    mavutil.mavlink.MAV_SEVERITY_INFO,
                )
                prev_state = STATE
            log.info("── STATE: %-30s MODE: %s", STATE, control.get_mode())

            if   STATE == "wait_for_auto_start":
                STATE = wait_for_auto_start()
            elif STATE == "wait_for_handoff":
                STATE = wait_for_handoff()
            elif STATE == "guided_handoff":
                STATE = guided_handoff()

            elif STATE == "search_for_buoy":
                STATE = search_for_buoy(buoy_detector)
            elif STATE == "center_on_buoy":
                STATE = center_on_buoy(buoy_detector)
            elif STATE == "hold_at_buoy":
                STATE = hold_at_buoy()

            elif STATE == "search_for_payload":
                STATE = search_for_payload(payload_detector)
            elif STATE == "center_payload_target":
                STATE = center_payload_target(payload_detector)
            elif STATE == "drop_payload":
                STATE = drop_payload(payload_detector)

            elif STATE == "return_to_launch":
                STATE = return_to_launch()

            elif STATE == "manual_override":
                STATE = manual_override(payload_detector)

            else:
                log.error("Unknown state: %s", STATE)
                _gcs(
                    "AIMM: bad state {}".format(STATE)[:50],
                    mavutil.mavlink.MAV_SEVERITY_ERROR,
                )
                break

        if STATE == "done":
            _gcs("AIMM: done", mavutil.mavlink.MAV_SEVERITY_NOTICE)

    except KeyboardInterrupt:
        log.info("Ctrl+C — stopping")
        _gcs("AIMM: stopped (user)", mavutil.mavlink.MAV_SEVERITY_WARNING)
        try:
            control.stop_drone()
        except Exception:
            pass

    except Exception as e:
        log.error("Mission failed: %s", e)
        log.error(traceback.format_exc())
        err = str(e).replace("\n", " ")
        _gcs(
            "AIMM: {}".format(err)[:50],
            mavutil.mavlink.MAV_SEVERITY_ERROR,
        )
        try:
            control.stop_drone()
        except Exception:
            pass

    finally:
        for det in [buoy_detector, payload_detector]:
            try:
                if det is not None:
                    det.close()
            except Exception:
                pass

        log.info("=== main_controller.py finished ===")


if __name__ == "__main__":
    main()
