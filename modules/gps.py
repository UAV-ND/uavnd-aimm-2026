# modules/gps.py
"""
Simple MAVLink GPS interface (Jetson Nano <-> Cube Orange) using pymavlink.

Goal:
- Provide a GPS object you can call to get the drone's current absolute position.
- Works for serial (Cube via USB/TELEM) and sim (UDP from SITL / simulator).

Reads these MAVLink messages when available:
- GLOBAL_POSITION_INT  (lat/lon/relative_alt/vx/vy/vz/hdg)
- GPS_RAW_INT          (fix_type, satellites_visible, eph/epv, alt)

Typical usage (serial):
    from modules import gps
    g = gps.GPS("/dev/ttyACM0", baud=57600)
    g.start()
    pos = g.get_position(timeout_s=2.0)
    print(pos)

Typical usage (sim):
    g = gps.GPS("127.0.0.1:14551")  # will auto-normalize to udp:127.0.0.1:14551
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any

from pymavlink import mavutil


# ----------------------------
# Data model
# ----------------------------

@dataclass
class GPSFix:
    """
    lat_deg/lon_deg are degrees.
    alt_m is altitude in meters (MSL when derived from GPS_RAW_INT, relative when derived from GLOBAL_POSITION_INT).
    """
    lat_deg: Optional[float] = None
    lon_deg: Optional[float] = None
    alt_m: Optional[float] = None

    # Extra info
    fix_type: Optional[int] = None           # 0..6 per MAVLink GPS fix types
    satellites_visible: Optional[int] = None
    eph_m: Optional[float] = None            # horizontal dilution/accuracy estimate (meters), if available
    epv_m: Optional[float] = None            # vertical accuracy estimate (meters), if available

    # Velocity (m/s), if available
    vx_m_s: Optional[float] = None
    vy_m_s: Optional[float] = None
    vz_m_s: Optional[float] = None

    # Heading (deg), if available
    heading_deg: Optional[float] = None

    # Timestamps
    time_boot_ms: Optional[int] = None
    last_update_monotonic: Optional[float] = None


# ----------------------------
# Helpers
# ----------------------------

def _normalize_connection(conn: str) -> str:
    """
    Accepts:
      - "/dev/ttyACM0"
      - "127.0.0.1:14551"
      - "udp:127.0.0.1:14551"
      - "tcp:127.0.0.1:5760"
      - "udpin:0.0.0.0:14551" etc.

    Returns a string suitable for mavutil.mavlink_connection.
    """
    conn = (conn or "").strip()
    if not conn:
        raise ValueError("Empty connection string")

    # If it looks like a path, keep it (serial)
    if conn.startswith("/dev/") or conn.startswith("COM") or conn.startswith("\\\\.\\"):
        return conn

    # If it already has a MAVLink prefix, keep it
    if conn.startswith(("udp:", "udpin:", "udpout:", "tcp:", "tcpin:", "tcpout:")):
        return conn

    # If it looks like host:port, assume UDP
    if ":" in conn and "/" not in conn:
        return f"udp:{conn}"

    # Fallback: pass through
    return conn


def _safe_float(x, scale: float = 1.0) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x) * scale
    except Exception:
        return None


# ----------------------------
# Main GPS class
# ----------------------------

class GPS:
    def __init__(
        self,
        connection: str,
        baud: int = 57600,
        source_system: int = 255,
        source_component: int = 0,
        wait_heartbeat_s: float = 10.0,
    ):
        """
        connection:
          Serial: "/dev/ttyACM0" (or /dev/ttyTHS1 etc.)
          Sim:    "127.0.0.1:14551"  (auto -> "udp:127.0.0.1:14551")
        """
        self.connection_in = connection
        self.connection = _normalize_connection(connection)
        self.baud = baud
        self.source_system = source_system
        self.source_component = source_component
        self.wait_heartbeat_s = wait_heartbeat_s

        self._mav: Optional[mavutil.mavfile] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        self._lock = threading.Lock()
        self._fix = GPSFix()
        self._last_hb_monotonic: Optional[float] = None

    # ---- lifecycle ----

    def start(self) -> None:
        """Open the MAVLink connection and start listener thread."""
        if self._thread and self._thread.is_alive():
            return

        # Create connection
        if self.connection.startswith("/dev/") or self.connection.startswith("COM") or self.connection.startswith("\\\\.\\"):
            self._mav = mavutil.mavlink_connection(
                self.connection,
                baud=self.baud,
                source_system=self.source_system,
                source_component=self.source_component,
            )
        else:
            self._mav = mavutil.mavlink_connection(
                self.connection,
                source_system=self.source_system,
                source_component=self.source_component,
            )

        # Wait for heartbeat so we know sysid/compid
        hb = self._mav.wait_heartbeat(timeout=self.wait_heartbeat_s)
        if hb is None:
            raise TimeoutError(f"No MAVLink heartbeat within {self.wait_heartbeat_s}s on {self.connection!r}")
        self._last_hb_monotonic = time.monotonic()

        # Ask for streams at a reasonable rate (works on ArduPilot)
        # These are best-effort; if unsupported it won't crash.
        try:
            # GLOBAL_POSITION_INT (msg id 33), GPS_RAW_INT (msg id 24)
            self._request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, 5.0)
            self._request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT, 5.0)
        except Exception:
            pass

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop listener thread and close connection."""
        self._stop_evt.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

        if self._mav is not None:
            try:
                self._mav.close()
            except Exception:
                pass
        self._mav = None

    # ---- public API ----

    def get_position(self, timeout_s: float = 0.0, require_fix: bool = False) -> GPSFix:
        """
        Returns the latest known GPSFix.

        timeout_s:
          If > 0, waits up to timeout_s for a fresh update that includes lat/lon.

        require_fix:
          If True, also requires fix_type >= 3 (3D fix) before returning (or until timeout).
        """
        deadline = time.monotonic() + max(0.0, timeout_s)

        while True:
            with self._lock:
                fix = GPSFix(**self._fix.__dict__)  # copy

            has_latlon = (fix.lat_deg is not None and fix.lon_deg is not None)
            has_3d_fix = (fix.fix_type is not None and fix.fix_type >= 3)

            ok = has_latlon and ((not require_fix) or has_3d_fix)
            if ok:
                return fix

            if timeout_s <= 0.0 or time.monotonic() >= deadline:
                return fix  # return best-effort (possibly None fields)

            time.sleep(0.05)

    def is_connected(self, stale_after_s: float = 3.0) -> bool:
        """True if we have seen a heartbeat recently."""
        if self._last_hb_monotonic is None:
            return False
        return (time.monotonic() - self._last_hb_monotonic) <= stale_after_s

    def as_dict(self) -> Dict[str, Any]:
        """Convenience for logging/JSON."""
        fix = self.get_position()
        return fix.__dict__.copy()

    # ---- internal ----

    def _request_message_interval(self, msg_id: int, hz: float) -> None:
        """
        Uses MAV_CMD_SET_MESSAGE_INTERVAL (ArduPilot supports this).
        Interval is in microseconds.
        """
        if self._mav is None:
            return

        interval_us = int(1_000_000 / max(0.1, hz))
        self._mav.mav.command_long_send(
            self._mav.target_system,
            self._mav.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            msg_id,
            interval_us,
            0, 0, 0, 0, 0,
        )

    def _listen_loop(self) -> None:
        assert self._mav is not None

        while not self._stop_evt.is_set():
            try:
                msg = self._mav.recv_match(blocking=True, timeout=0.5)
            except Exception:
                continue

            if msg is None:
                continue

            mtype = msg.get_type()
            now = time.monotonic()

            if mtype == "HEARTBEAT":
                self._last_hb_monotonic = now
                continue

            # GLOBAL_POSITION_INT gives lat/lon in 1E7 deg, relative_alt in mm, vx/vy/vz in cm/s, hdg in cdeg.
            if mtype == "GLOBAL_POSITION_INT":
                lat = getattr(msg, "lat", None)
                lon = getattr(msg, "lon", None)
                rel_alt_mm = getattr(msg, "relative_alt", None)
                vx = getattr(msg, "vx", None)
                vy = getattr(msg, "vy", None)
                vz = getattr(msg, "vz", None)
                hdg = getattr(msg, "hdg", None)
                tboot = getattr(msg, "time_boot_ms", None)

                with self._lock:
                    if lat is not None and lon is not None and int(lat) != 0 and int(lon) != 0:
                        self._fix.lat_deg = int(lat) / 1e7
                        self._fix.lon_deg = int(lon) / 1e7

                    if rel_alt_mm is not None:
                        self._fix.alt_m = int(rel_alt_mm) / 1000.0  # relative altitude (m)

                    # velocities (cm/s -> m/s)
                    if vx is not None:
                        self._fix.vx_m_s = int(vx) / 100.0
                    if vy is not None:
                        self._fix.vy_m_s = int(vy) / 100.0
                    if vz is not None:
                        self._fix.vz_m_s = int(vz) / 100.0

                    # heading (cdeg -> deg). 65535 means unknown.
                    if hdg is not None and int(hdg) != 65535:
                        self._fix.heading_deg = int(hdg) / 100.0

                    if tboot is not None:
                        self._fix.time_boot_ms = int(tboot)

                    self._fix.last_update_monotonic = now

                continue

            # GPS_RAW_INT gives fix_type, satellites, eph/epv (cm), alt (mm).
            if mtype == "GPS_RAW_INT":
                fix_type = getattr(msg, "fix_type", None)
                sats = getattr(msg, "satellites_visible", None)
                eph = getattr(msg, "eph", None)
                epv = getattr(msg, "epv", None)
                alt_mm = getattr(msg, "alt", None)
                lat = getattr(msg, "lat", None)
                lon = getattr(msg, "lon", None)
                tboot = getattr(msg, "time_usec", None)  # microseconds since boot (often)

                with self._lock:
                    if fix_type is not None:
                        self._fix.fix_type = int(fix_type)
                    if sats is not None:
                        self._fix.satellites_visible = int(sats)

                    # eph/epv are in cm per MAVLink; some firmwares use 65535 as unknown
                    if eph is not None and int(eph) != 65535:
                        self._fix.eph_m = int(eph) / 100.0
                    if epv is not None and int(epv) != 65535:
                        self._fix.epv_m = int(epv) / 100.0

                    if alt_mm is not None and int(alt_mm) != 0:
                        # This is GPS altitude above MSL (meters)
                        self._fix.alt_m = int(alt_mm) / 1000.0

                    if lat is not None and lon is not None and int(lat) != 0 and int(lon) != 0:
                        self._fix.lat_deg = int(lat) / 1e7
                        self._fix.lon_deg = int(lon) / 1e7

                    # Keep time_boot_ms if we can map it; otherwise just store monotonic timestamp
                    if tboot is not None:
                        # convert usec -> ms (fits in int)
                        self._fix.time_boot_ms = int(int(tboot) / 1000)

                    self._fix.last_update_monotonic = now

                continue


# ----------------------------
# Optional convenience singleton (similar style to your other modules)
# ----------------------------

_gps_singleton: Optional[GPS] = None

def connect_gps(connection: str, baud: int = 57600) -> GPS:
    """
    Convenience: create + start a singleton GPS instance.
    """
    global _gps_singleton
    _gps_singleton = GPS(connection, baud=baud)
    _gps_singleton.start()
    return _gps_singleton

def get_gps() -> GPS:
    if _gps_singleton is None:
        raise RuntimeError("GPS not connected. Call connect_gps(...) first.")
    return _gps_singleton

def get_position(timeout_s: float = 0.0, require_fix: bool = False) -> GPSFix:
    """
    Convenience: get position from singleton.
    """
    return get_gps().get_position(timeout_s=timeout_s, require_fix=require_fix)
