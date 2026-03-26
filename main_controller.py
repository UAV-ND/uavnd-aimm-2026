import sys, time, logging, datetime, argparse
sys.path.insert(1, '/home/uav-nano/Documents/aimm-dev/uavnd-aimm-2026/modules')

import cv2
import collections

import drone
import lidar
import detector_mobilenet as detector
import vision
import control
import keyboard

# --- Args ---
parser = argparse.ArgumentParser(description='AIMM Main Controller')
parser.add_argument('--debug_path', type=str, default="debug/run1", help='debug output path')
parser.add_argument('--mode', type=str, default='flight', help='flight or test')
parser.add_argument('--control', type=str, default='PID', help='PID or P controller')
args = parser.parse_args()

# --- Logging Setup (unique file per run) ---
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = f'/home/uav-nano/Documents/aimm-dev/uavnd-aimm-2026/test/mission_{timestamp}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger()
log.info("=== main_controller.py started ===")
log.info(f"Log file: {log_path}")

# --- Mission Config ---
MAX_ALT = 2.5               # meters
MAX_FOLLOW_DIST = 2         # meters
MAX_MA_X_LEN = 5
MAX_MA_Z_LEN = 5
MA_X = collections.deque(maxlen=MAX_MA_X_LEN)
MA_Z = collections.deque(maxlen=MAX_MA_Z_LEN)

# GPS waypoints (TODO: set real coords)
BUOY_GPS = (0.0, 0.0, MAX_ALT)       # lat, lon, alt
BOAT_GPS = (0.0, 0.0, MAX_ALT)       # lat, lon, alt

# Timeouts (seconds)
SEARCH_TIMEOUT = 40
DESCENT_MARKER_LOST_TIMEOUT = 5

# --- Setup ---
def setup():
    log.info("Connecting lidar...")
    lidar.connect_lidar("/dev/ttyTHS1")

    log.info("Setting up detector...")
    detector.initialize_detector()

    log.info("Connecting to drone...")
    if args.mode == "flight":
        log.info("MODE = flight")
        control.connect_drone('/dev/ttyACM0')
    else:
        log.info("MODE = test")
        control.connect_drone('127.0.0.1:14551')

    control.set_flight_altitude(MAX_ALT)
    control.configure_PID(args.control)
    control.initialize_debug_logs(args.debug_path)

    log.info("Setup complete")


# --- Abort Check (runs every state tick) ---
def check_abort():
    """Returns True if mission should abort. Check RC override, failsafe, manual kill."""
    if keyboard.is_pressed('q'):
        log.warning("Manual abort triggered (keyboard)")
        return True
    # TODO: check RC override channel
    # TODO: check drone failsafe flags
    return False


# ============================================================
#  STATE HANDLERS
#  Each returns the next state name as a string.
# ============================================================

def state_takeoff():
    log.info("STATE: takeoff")
    control.print_drone_report()
    control.arm_and_takeoff(MAX_ALT)
    log.info("Takeoff complete")
    return "goto_buoy_gps"


def state_goto_buoy_gps():
    log.info("STATE: goto_buoy_gps")
    pass  # TODO: implement
    return "search_buoy"


def state_search_buoy():
    log.info("STATE: search_buoy")
    pass  # TODO: implement
    return "center_buoy"


def state_center_buoy():
    log.info("STATE: center_buoy")
    pass  # TODO: implement (recovery: return "search_buoy" if buoy lost)
    return "search_target"


def state_search_target():
    log.info("STATE: search_target")
    pass  # TODO: implement
    return "center_target"


def state_center_target():
    log.info("STATE: center_target")
    pass  # TODO: implement (recovery: return "search_target" if target lost)
    return "payload_drop"


def state_payload_drop():
    log.info("STATE: payload_drop")
    pass  # TODO: implement
    return "goto_boat_gps"


def state_goto_boat_gps():
    log.info("STATE: goto_boat_gps")
    pass  # TODO: implement
    return "search_landing_marker"


def state_search_landing_marker():
    log.info("STATE: search_landing_marker")
    pass  # TODO: implement
    return "center_landing_marker"


def state_center_landing_marker():
    log.info("STATE: center_landing_marker")
    pass  # TODO: implement (recovery: return "search_landing_marker" if marker lost)
    return "controlled_descent"


def state_controlled_descent():
    log.info("STATE: controlled_descent")
    pass  # TODO: implement (recovery: return "search_landing_marker" if marker lost during descent)
    return "land"


def state_land():
    log.info("STATE: land")
    control.land()
    detector.close_camera()
    log.info("Landed successfully")
    return "done"


def state_abort():
    log.error("STATE: abort — emergency landing")
    try:
        control.land()
    except Exception as e:
        log.error(f"Abort landing failed: {e}")
    try:
        detector.close_camera()
    except Exception:
        pass
    return "done"


# --- State Table ---
STATE_TABLE = {
    "takeoff":                state_takeoff,
    "goto_buoy_gps":         state_goto_buoy_gps,
    "search_buoy":           state_search_buoy,
    "center_buoy":           state_center_buoy,
    "search_target":         state_search_target,
    "center_target":         state_center_target,
    "payload_drop":          state_payload_drop,
    "goto_boat_gps":         state_goto_boat_gps,
    "search_landing_marker": state_search_landing_marker,
    "center_landing_marker": state_center_landing_marker,
    "controlled_descent":    state_controlled_descent,
    "land":                  state_land,
    "abort":                 state_abort,
}


# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    setup()

    # --- Wait for RC arm ---
    log.info("Waiting for drone to be armed via RC...")
    while not control.vehicle.armed:
        log.info("Not armed yet, waiting...")
        time.sleep(1)
    log.info("Drone is ARMED — proceeding with mission")

    # --- State Machine ---
    current_state = "takeoff"

    while current_state != "done":
        handler = STATE_TABLE.get(current_state)
        if handler is None:
            log.error(f"Unknown state: {current_state} — aborting")
            current_state = "abort"
            continue

        try:
            next_state = handler()
            log.info(f"Transition: {current_state} -> {next_state}")
            current_state = next_state
        except Exception as e:
            log.error(f"Exception in state '{current_state}': {e}", exc_info=True)
            current_state = "abort"

    log.info("=== main_controller.py finished ===")
