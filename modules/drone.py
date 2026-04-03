from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
from pymavlink import mavutil
import math

vehicle = None


def connect_drone(connection_string, waitready=True, baudrate=None):
    global vehicle

    vehicle = connect(
        connection_string,
        wait_ready=False,
        baud=baudrate,
        heartbeat_timeout=60
    )

    try:
        vehicle.wait_ready('mode', 'armed', 'attitude', 'location', 'commands', timeout=120)
    except Exception as e:
        print("[WARN] wait_ready minimal timed out: {}".format(e))

    print("drone connected")
    return vehicle


def disconnect_drone():
    global vehicle
    if vehicle is not None:
        vehicle.close()
        vehicle = None


def get_version():
    global vehicle
    return vehicle.version


def get_location():
    global vehicle
    return vehicle.location.global_frame


def get_location_relative():
    global vehicle
    return vehicle.location.global_relative_frame


def get_altitude():
    global vehicle
    return vehicle.location.global_relative_frame.alt


def get_velocity():
    global vehicle
    return vehicle.velocity


def get_battery_info():
    global vehicle
    return vehicle.battery


def get_mode():
    global vehicle
    return vehicle.mode.name


def get_home_location():
    global vehicle
    return vehicle.home_location


def get_heading():
    global vehicle
    return vehicle.heading


def get_EKF_status():
    global vehicle
    return vehicle.ekf_ok


def get_ground_speed():
    global vehicle
    return vehicle.groundspeed


def read_channel(channel):
    global vehicle
    return vehicle.channels[str(channel)]


def set_groundspeed(speed):
    global vehicle
    print("groundspeed set to: {}".format(speed))
    vehicle.groundspeed = speed


def set_flight_mode(f_mode):
    global vehicle
    print("Setting mode to {}".format(f_mode))
    vehicle.mode = VehicleMode(f_mode)


def set_param(param, value):
    global vehicle
    vehicle.parameters[param] = value


def get_param(param):
    global vehicle
    return vehicle.parameters[param]


def set_channel(channel, value):
    global vehicle
    vehicle.channels.overrides[str(channel)] = value


def clear_channel(channel):
    global vehicle
    vehicle.channels.overrides[str(channel)] = None


def get_channel_override(channel):
    global vehicle
    return vehicle.channels.overrides[str(channel)]


def disarm():
    global vehicle
    vehicle.armed = False


def arm():
    global vehicle
    vehicle.groundspeed = 3

    print("Basic pre-arm checks")
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("ARMED")


def arm_and_takeoff(aTargetAltitude):
    global vehicle

    print("setting groundspeed to 3")
    vehicle.groundspeed = 3

    print("Basic pre-arm checks")
    print("System status:", vehicle.system_status.state)
    print("EKF ok:", getattr(vehicle, "ekf_ok", None))
    print("GPS fix:", vehicle.gps_0.fix_type, "sats:", vehicle.gps_0.satellites_visible)
    print("Last heartbeat:", vehicle.last_heartbeat)

    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        print(" is_armable:", vehicle.is_armable,
              "| system_status:", vehicle.system_status.state,
              "| ekf_ok:", getattr(vehicle, "ekf_ok", None),
              "| gps_fix:", vehicle.gps_0.fix_type,
              "| sats:", vehicle.gps_0.satellites_visible)
        time.sleep(1)

    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)

    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(" Altitude: ", alt)
        if alt >= aTargetAltitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)


def land():
    global vehicle
    print("Setting LAND mode...")
    vehicle.mode = VehicleMode("LAND")


def return_to_launch_location():
    global vehicle
    print("Setting RTL mode...")
    vehicle.mode = VehicleMode("RTL")


def goto_location(lat, lon, alt, groundspeed=1.5):
    global vehicle
    target = LocationGlobalRelative(lat, lon, alt)
    print("simple_goto -> lat={}, lon={}, alt={}, groundspeed={}".format(lat, lon, alt, groundspeed))
    vehicle.simple_goto(target, groundspeed=groundspeed)


def get_distance_metres(lat1, lon1, lat2, lon2):
    dlat = (lat2 - lat1) * 111320.0
    dlon = (lon2 - lon1) * 111320.0 * math.cos(math.radians(lat1))
    return math.sqrt((dlat * dlat) + (dlon * dlon))


def distance_to_waypoint(lat, lon):
    global vehicle
    loc = vehicle.location.global_relative_frame
    return get_distance_metres(loc.lat, loc.lon, lat, lon)


def send_movement_command_YAW(heading):
    global vehicle
    speed = 0
    direction = 1

    print("Sending YAW movement command with heading: %f" % heading)

    if heading < 0:
        heading = heading * -1
        direction = -1

    msg = vehicle.message_factory.command_long_encode(
        0, 0,
        mavutil.mavlink.MAV_CMD_CONDITION_YAW,
        0,
        heading,
        speed,
        direction,
        1,
        0, 0, 0
    )

    vehicle.send_mavlink(msg)


def send_movement_command_XYZ(velocity_x, velocity_y, altitude):
    """
    BODY_NED command
    velocity_x: +forward / -back
    velocity_y: +right / -left
    """
    global vehicle

    print("Sending XYZ movement command with v_x=%f v_y=%f altitude=%f" % (velocity_x, velocity_y, altitude))

    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        0b0000111111100011,
        0, 0, altitude,
        velocity_x, velocity_y, 0,
        0, 0, 0,
        0, 0
    )

    vehicle.send_mavlink(msg)


def send_body_velocity_xy(velocity_forward, velocity_right, altitude):
    send_movement_command_XYZ(velocity_forward, velocity_right, altitude)


def stop_body_motion(altitude):
    send_movement_command_XYZ(0, 0, altitude)


def download_mission():
    global vehicle
    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready()
    return cmds


def list_mission_commands():
    cmds = download_mission()
    items = []
    for cmd in cmds:
        items.append(cmd)
    return items


def print_mission():
    cmds = list_mission_commands()
    print("===== Uploaded Mission =====")
    for cmd in cmds:
        print("seq={} command={} lat={} lon={} alt={}".format(
            cmd.seq, cmd.command, cmd.x, cmd.y, cmd.z
        ))


def get_mission_waypoint_by_index(index):
    cmds = list_mission_commands()

    if index < 0 or index >= len(cmds):
        raise IndexError("Mission index {} out of range; mission has {} items".format(index, len(cmds)))

    cmd = cmds[index]

    if cmd.command != mavutil.mavlink.MAV_CMD_NAV_WAYPOINT:
        raise RuntimeError("Mission item {} is not NAV_WAYPOINT, command={}".format(index, cmd.command))

    return {
        "seq": cmd.seq,
        "lat": cmd.x,
        "lon": cmd.y,
        "alt": cmd.z,
        "command": cmd.command
    }


def get_next_mission_index():
    global vehicle
    return getattr(vehicle.commands, "next", None)