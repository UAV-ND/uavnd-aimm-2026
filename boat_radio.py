#!/usr/bin/env python3
# boat_radio.py
#
# Receives boat GPS from the FC via MAVLink NAMED_VALUE_FLOAT messages.
#
# The Lua script (boat_gps.lua) running on ArduPilot reads the plain ASCII
# "BOAT,lat,lon" string off TELEM1, parses it, and injects two messages:
#   NAMED_VALUE_FLOAT  name="BOAT_LAT"  value=<lat>
#   NAMED_VALUE_FLOAT  name="BOAT_LON"  value=<lon>
#
# MAVProxy forwards these to the Jetson UDP stream. This module listens on
# the existing DroneKit vehicle connection — no extra serial port needed.
#
# Usage:
#   import boat_radio
#   boat_radio.connect_boat_radio(vehicle)   # pass DroneKit vehicle object
#   gps = boat_radio.read_boat_gps()         # returns dict or None
#   boat_radio.disconnect_boat_radio()

import time
import threading
from dronekit import Vehicle

# How long (seconds) a fix is considered fresh before being discarded.
# If only one of lat/lon arrives and no matching pair comes within this
# window, the partial fix is dropped.
PAIR_TIMEOUT_S = 2.0

_vehicle       = None
_listener_lock = threading.Lock()

# Latest complete GPS fix {"lat": float, "lon": float, "timestamp": float}
_latest_gps    = None

# Partial fix storage — holds a lat or lon waiting for its partner
_pending_lat   = None
_pending_lon   = None
_pending_ts    = 0.0


def _on_named_value_float(self, name, message):
    """
    DroneKit message listener for NAMED_VALUE_FLOAT.
    Called on the DroneKit background thread — all shared state is locked.
    """
    global _latest_gps, _pending_lat, _pending_lon, _pending_ts

    msg_name  = message.name.rstrip('\x00')   # MAVLink pads with null bytes
    msg_value = message.value
    now       = time.time()

    with _listener_lock:
        # Discard stale partial fixes
        if now - _pending_ts > PAIR_TIMEOUT_S:
            _pending_lat = None
            _pending_lon = None

        if msg_name == "BOAT_LAT":
            _pending_lat = msg_value
            _pending_ts  = now

        elif msg_name == "BOAT_LON":
            _pending_lon = msg_value
            _pending_ts  = now

        else:
            return   # not a boat message — ignore

        # If we have both halves of the pair, commit the fix
        if _pending_lat is not None and _pending_lon is not None:
            lat = _pending_lat
            lon = _pending_lon

            # Basic sanity check
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                _latest_gps  = {"lat": lat, "lon": lon, "timestamp": now}
            else:
                print("[boat_radio] WARN: received out-of-range coords "
                      "lat={} lon={} — discarding".format(lat, lon))

            # Reset pending regardless of validity
            _pending_lat = None
            _pending_lon = None


def connect_boat_radio(vehicle):
    """
    Register the NAMED_VALUE_FLOAT listener on an existing DroneKit vehicle.
    Call this once after connecting to the drone.

    Args:
        vehicle: a connected dronekit.Vehicle object
    """
    global _vehicle
    _vehicle = vehicle
    _vehicle.add_message_listener("NAMED_VALUE_FLOAT", _on_named_value_float)
    print("[boat_radio] Listening for BOAT_LAT / BOAT_LON on MAVLink stream")


def disconnect_boat_radio():
    """Remove the MAVLink listener. Safe to call even if never connected."""
    global _vehicle, _latest_gps, _pending_lat, _pending_lon
    if _vehicle is not None:
        try:
            _vehicle.remove_message_listener("NAMED_VALUE_FLOAT", _on_named_value_float)
        except Exception:
            pass
        _vehicle = None

    with _listener_lock:
        _latest_gps  = None
        _pending_lat = None
        _pending_lon = None

    print("[boat_radio] Disconnected")


def read_boat_gps():
    """
    Return the latest complete boat GPS fix, or None if no fix has arrived.

    Returns:
        dict {"lat": float, "lon": float, "timestamp": float}  or  None
    """
    with _listener_lock:
        return _latest_gps