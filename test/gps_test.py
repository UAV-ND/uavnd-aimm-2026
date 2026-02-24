#!/usr/bin/env python3
# test GPS link between drone and jetson nano (MAVLink)
# - Gracefully exits with clear logs if dependencies fail (missing modules, port not found, no heartbeat, etc.)

import os
import sys
import time
import traceback


# ---- config ----
MODULES_PATH = "/home/uav-nano/Documents/aimm-dev/uavnd-aimm-2026/modules"

CONNECTION = "/dev/ttyACM0"   # Cube Orange over USB serial
BAUD = 57600                 # change if your setup differs (ex: 921600)
REQUIRE_3D_FIX = False        # set True to wait for fix_type >= 3
PRINT_HZ = 2                  # how many prints per second

LOG_DIR = "/home/uav-nano/Documents/aimm-dev/uavnd-aimm-2026/test/logs"
LOG_PREFIX = "gps_test"
# ----------------


def _now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg, level="INFO", logfile=None):
    line = f"[{_now_str()}] [{level}] {msg}"
    print(line)
    if logfile:
        try:
            logfile.write(line + "\n")
            logfile.flush()
        except Exception:
            # Don't crash due to logging issues
            pass


def graceful_exit(code, reason, logfile=None, exc=None):
    log(reason, level="ERROR", logfile=logfile)
    if exc is not None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log(tb, level="ERROR", logfile=logfile)
    raise SystemExit(code)


def main():
    # Ensure logs directory exists
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except Exception:
        # If we can't create logs, we still run (stdout only)
        pass

    log_path = os.path.join(LOG_DIR, f"{LOG_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logfile = None
    try:
        logfile = open(log_path, "a")
    except Exception:
        logfile = None

    log("Starting GPS test!", logfile=logfile)
    log(f"modules_path={MODULES_PATH}", logfile=logfile)
    log(f"connection={CONNECTION} baud={BAUD} require_3d_fix={REQUIRE_3D_FIX} print_hz={PRINT_HZ}", logfile=logfile)
    if logfile:
        log(f"log_file={log_path}", logfile=logfile)

    # Add modules path
    if not os.path.isdir(MODULES_PATH):
        graceful_exit(
            2,
            f"Modules path not found: {MODULES_PATH}. Fix MODULES_PATH in this script.",
            logfile=logfile,
        )
    sys.path.insert(1, MODULES_PATH)

    # Check serial device exists (helps catch 'Cube/Orange not found' early)
    if CONNECTION.startswith("/dev/") and not os.path.exists(CONNECTION):
        graceful_exit(
            3,
            f"Device not found: {CONNECTION}. Is the Cube Orange plugged in and visible? "
            f"Try: ls -l {CONNECTION}  (or check dmesg / /dev/ttyACM*)",
            logfile=logfile,
        )

    # Import gps module with clear dependency error messages
    try:
        import gps
    except ModuleNotFoundError as e:
        # Common missing deps: pymavlink, dataclasses (py3.6 backport)
        missing = getattr(e, "name", None) or str(e)
        hint = ""
        if missing == "pymavlink":
            hint = "Install with: pip3 install pymavlink"
        elif missing == "dataclasses":
            hint = "You're on Python 3.6. Install backport: pip3 install dataclasses  (or upgrade Python to 3.7+)"
        graceful_exit(
            4,
            f"Failed to import gps module due to missing dependency: {missing}. {hint}".strip(),
            logfile=logfile,
            exc=e,
        )
    except Exception as e:
        graceful_exit(5, "Failed to import gps module due to an unexpected error.", logfile=logfile, exc=e)

    g = None
    try:
        # Connect singleton (this is where heartbeat timeouts or permission errors show up)
        log("Connecting GPS singleton...", logfile=logfile)
        g = gps.connect_gps(CONNECTION, baud=BAUD)
        log("Connected. Waiting for GPS updates...\n", logfile=logfile)

        # Main print loop
        no_data_seconds = 0.0
        last_loop = time.monotonic()

        while True:
            pos = gps.get_position(timeout_s=1.0, require_fix=REQUIRE_3D_FIX)

            now = time.monotonic()
            dt = now - last_loop
            last_loop = now

            # Track whether we're seeing fresh GPS updates
            if pos.last_update_monotonic is None:
                no_data_seconds += dt
            else:
                # If last update is old, treat as no-data accumulating
                age = now - pos.last_update_monotonic
                if age > 2.5:
                    no_data_seconds += dt
                else:
                    no_data_seconds = 0.0

            if pos.last_update_monotonic is not None:
                age = now - pos.last_update_monotonic
            else:
                age = None

            # Print a readable block
            print("--------------------------------------------------")
            print(f"connected: {g.is_connected()}")
            print(f"lat (deg): {pos.lat_deg}")
            print(f"lon (deg): {pos.lon_deg}")
            print(f"alt (m):   {pos.alt_m}")
            print(f"fix type:  {pos.fix_type}")
            print(f"sats:      {pos.satellites_visible}")
            print(f"velocity:  vx={pos.vx_m_s}, vy={pos.vy_m_s}, vz={pos.vz_m_s}")
            print(f"heading:   {pos.heading_deg}")
            print(f"age (s):   {age}")
            print("--------------------------------------------------\n")

            # If we haven't gotten any usable updates for a while, exit with a helpful message
            if no_data_seconds >= 15.0:
                graceful_exit(
                    6,
                    "No fresh GPS/MAVLink position updates for ~15s. "
                    "Possible causes: no heartbeat, wrong port/baud, another process is using the port, "
                    "GPS not providing data, or stream rates not enabled.",
                    logfile=logfile,
                )

            time.sleep(1.0 / max(1, PRINT_HZ))

    except KeyboardInterrupt:
        log("Stopping GPS test due to keyboard interrupt.", level="INFO", logfile=logfile)
    except Exception as e:
        graceful_exit(7, "GPS test crashed due to an unexpected runtime error.", logfile=logfile, exc=e)
    finally:
        # Always stop gps cleanly
        if g is not None:
            try:
                g.stop()
            except Exception as e:
                log(f"Failed to stop GPS cleanly: {e}", level="WARN", logfile=logfile)

        if logfile:
            try:
                logfile.close()
            except Exception:
                pass

        print("Finished.")


if __name__ == "__main__":
    main()
