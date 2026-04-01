from modules import drone
from simple_pid import PID
import time

MAX_VEL_FORWARD = 1.0
MAX_VEL_RIGHT = 1.0

P_X = 0.008
I_X = 0.0
D_X = 0.0

P_Y = 0.008
I_Y = 0.0
D_Y = 0.0

pid_x = None
pid_y = None

input_err_x = 0.0
input_err_y = 0.0

cmd_vel_forward = 0.0
cmd_vel_right = 0.0

flight_altitude = 4.0
state = "idle"


def configure_PID(control="PID"):
    global pid_x, pid_y

    if control == "PID":
        pid_x = PID(P_X, I_X, D_X, setpoint=0)
        pid_y = PID(P_Y, I_Y, D_Y, setpoint=0)
    else:
        pid_x = PID(P_X, 0, 0, setpoint=0)
        pid_y = PID(P_Y, 0, 0, setpoint=0)

    pid_x.output_limits = (-MAX_VEL_RIGHT, MAX_VEL_RIGHT)
    pid_y.output_limits = (-MAX_VEL_FORWARD, MAX_VEL_FORWARD)

    print("Configured control:", control)


def connect_drone(drone_location):
    return drone.connect_drone(drone_location)


def get_vehicle():
    return drone.vehicle


def get_mode():
    return drone.get_mode()


def set_system_state(current_state):
    global state
    state = current_state


def set_flight_altitude(alt):
    global flight_altitude
    flight_altitude = alt


def wait_until_armed():
    print("Waiting for drone to be armed via RC...")
    while not drone.vehicle.armed:
        print("Not armed yet, waiting...")
        time.sleep(1)
    print("Drone is ARMED")


def wait_for_rc_start(channel='6', threshold=1800):
    print("Waiting for RC mission start on channel", channel)
    while True:
        try:
            val = drone.vehicle.channels.get(channel)
            if val is not None:
                val = int(val)
                print("RC channel {} = {}".format(channel, val))
                if val >= threshold:
                    print("RC mission start detected")
                    return True
        except Exception as e:
            print("RC read error:", e)

        time.sleep(0.25)


def pilot_took_over():
    """
    If the pilot flips to LOITER, autonomy should stop.
    """
    try:
        mode = drone.get_mode()
        if mode == "LOITER":
            print("Pilot takeover detected: mode is LOITER")
            return True
        return False
    except Exception as e:
        print("pilot_took_over check error:", e)
        return False


def get_target_waypoint_from_mission(use_waypoint_mode="first_nav", explicit_index=None):
    if use_waypoint_mode == "first_nav":
        wp = drone.get_first_nav_waypoint()
    elif use_waypoint_mode == "second_nav":
        wp = drone.get_second_nav_waypoint()
    elif use_waypoint_mode == "by_index":
        if explicit_index is None:
            raise RuntimeError("explicit_index required for by_index mode")
        wp = drone.get_mission_waypoint_by_index(explicit_index)
    else:
        raise RuntimeError("Unknown waypoint mode: {}".format(use_waypoint_mode))

    print("Selected mission waypoint:", wp)
    return wp


def setXdelta(x_delta):
    global input_err_x
    input_err_x = x_delta


def setYdelta(y_delta):
    global input_err_y
    input_err_y = y_delta


def get_forward_velocity():
    return cmd_vel_forward


def get_right_velocity():
    return cmd_vel_right


def arm_and_takeoff(max_height):
    drone.arm_and_takeoff(max_height)


def land():
    drone.land()


def rtl():
    drone.return_to_launch_location()


def print_drone_report():
    print(drone.get_EKF_status())
    print(drone.get_battery_info())
    print(drone.get_version())
    print(drone.get_mode())
    print(drone.get_location())


def goto_gps_location(lat, lon, alt, groundspeed=1.5, radius_m=1.5, timeout_s=90):
    drone.goto_location(lat, lon, alt, groundspeed=groundspeed)

    start = time.time()
    while time.time() - start < timeout_s:
        if pilot_took_over():
            return False

        dist = drone.distance_to_waypoint(lat, lon)
        print("Distance to waypoint:", round(dist, 2), "m")
        if dist <= radius_m:
            print("Reached waypoint")
            return True
        time.sleep(1.0)

    print("goto_gps_location timeout")
    return False


def stop_drone():
    global cmd_vel_forward, cmd_vel_right
    cmd_vel_forward = 0.0
    cmd_vel_right = 0.0
    drone.stop_body_motion(flight_altitude)


def hold_position(seconds=1.0):
    start = time.time()
    while time.time() - start < seconds:
        if pilot_took_over():
            return False
        stop_drone()
        time.sleep(0.1)
    return True


def control_drone():
    """
    Downward camera centering:
    x error -> left/right motion
    y error -> forward/back motion

    You may need to flip signs after first test.
    """
    global cmd_vel_forward, cmd_vel_right

    if abs(input_err_x) < 1e-6:
        cmd_vel_right = 0.0
    else:
        cmd_vel_right = -pid_x(input_err_x)

    if abs(input_err_y) < 1e-6:
        cmd_vel_forward = 0.0
    else:
        cmd_vel_forward = -pid_y(input_err_y)

    drone.send_body_velocity_xy(cmd_vel_forward, cmd_vel_right, flight_altitude)