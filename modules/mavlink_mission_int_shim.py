# mavlink_mission_int_shim.py
#
# DroneKit uses waypoint_request_send() -> legacy MISSION_REQUEST; ArduPilot 4.x
# prefers MISSION_REQUEST_INT and answers with MISSION_ITEM_INT. Stock DroneKit
# does not handle MISSION_ITEM_INT, so we patch the link and mirror the stock
# mission-download listener (see dronekit Vehicle MISSION_ITEM handler).

from dronekit import LocationGlobal
from pymavlink import mavutil


def _mission_item_int_to_mission_item(msg):
    """Global missions from MP use lat/lon as x/y * 1e7 (see pymavlink tools/mavmission.py)."""
    try:
        return mavutil.mavlink.MAVLink_mission_item_message(
            msg.target_system,
            msg.target_component,
            msg.seq,
            msg.frame,
            msg.command,
            msg.current,
            msg.autocontinue,
            msg.param1,
            msg.param2,
            msg.param3,
            msg.param4,
            msg.x * 1.0e-7,
            msg.y * 1.0e-7,
            float(msg.z),
            getattr(msg, "mission_type", 0),
        )
    except TypeError:
        return mavutil.mavlink.MAVLink_mission_item_message(
            msg.target_system,
            msg.target_component,
            msg.seq,
            msg.frame,
            msg.command,
            msg.current,
            msg.autocontinue,
            msg.param1,
            msg.param2,
            msg.param3,
            msg.param4,
            msg.x * 1.0e-7,
            msg.y * 1.0e-7,
            float(msg.z),
        )


def install_mission_int_shim(vehicle):
    """
    Patch vehicle._master.waypoint_request_send to use MISSION_REQUEST_INT and
    register MISSION_ITEM_INT handling so mission download still completes.
    Safe to call once per Vehicle; no-op if pymavlink lacks mission_request_int_send.
    """
    if vehicle is None or getattr(vehicle, "_aimm_mission_int_shim", False):
        return True
    master = vehicle._master
    if not hasattr(master.mav, "mission_request_int_send"):
        print("[WARN] pymavlink has no mission_request_int_send; MISSION_INT shim skipped")
        return False

    def waypoint_request_send_int(seq):
        seq = int(seq)
        ts, tc = master.target_system, master.target_component
        try:
            master.mav.mission_request_int_send(ts, tc, seq)
        except TypeError:
            master.mav.mission_request_int_send(ts, tc, seq, 0)

    master.waypoint_request_send = waypoint_request_send_int

    def on_mission_item_int(veh, name, msg):
        if veh._wp_loaded:
            return
        m = _mission_item_int_to_mission_item(msg)
        if m.seq == 0:
            if not (m.x == 0 and m.y == 0 and m.z == 0):
                veh._home_location = LocationGlobal(m.x, m.y, m.z)
        if m.seq > veh._wploader.count():
            pass
        elif m.seq < veh._wploader.count():
            pass
        else:
            veh._wploader.add(m)

        if m.seq + 1 < veh._wploader.expected_count:
            waypoint_request_send_int(m.seq + 1)
        else:
            veh._wp_loaded = True
            veh.notify_attribute_listeners("commands", veh.commands)

    vehicle.add_message_listener("MISSION_ITEM_INT", on_mission_item_int)
    vehicle._aimm_mission_int_shim = True
    print("MISSION_INT shim installed (MISSION_REQUEST_INT / MISSION_ITEM_INT)")
    return True
