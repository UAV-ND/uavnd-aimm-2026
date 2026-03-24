import sys, time, logging
sys.path.insert(1,'/home/uav-nano/Documents/aimm-dev/uavnd-aimm-2026/modules')
import drone

# --- Logging Setup ---
log_path = '/home/uav-nano/Documents/aimm-dev/uavnd-aimm-2026/test/drone_test.log'
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
log.info("=== drone_test.py started ===")

# config
height = 1
speed = 3   # m/s
size = 165  # 10 meter

log.info("Connecting to drone...")
v = drone.connect_drone('udpin:127.0.0.1:14552')
log.info(f"Connected: {v is not None}")
log.info(f"Mode: {v.mode}")
log.info(f"Mode: {drone.vehicle.mode if hasattr(drone, 'vehicle') else 'n/a'}")

# --- Wait for RC arm ---
log.info("Waiting for drone to be armed via RC...")
while not drone.vehicle.armed:
    log.info("Not armed yet, waiting...")
    time.sleep(1)
log.info("Drone is ARMED — proceeding with flight script")

log.info(f"Arming and taking off to height: {height}m")
drone.arm_and_takeoff(height)

log.info(f"Version: {drone.get_version()}")
log.info(f"Mode: {drone.get_mode()}")
log.info(f"Location: {drone.get_location()}")

time.sleep(5)

# fly rectangle with XYZ
log.info("Starting rectangle flight (XYZ)...")
for i in range(size):
    drone.send_movement_command_XYZ(speed,0,0)
    time.sleep(0.02)
log.info("Leg 1 complete (+X)")
time.sleep(5)

for i in range(size):
    drone.send_movement_command_XYZ(0,speed,0)
    time.sleep(0.02)
log.info("Leg 2 complete (+Y)")
time.sleep(5)

for i in range(size):
    drone.send_movement_command_XYZ(-speed,0,0)
    time.sleep(0.02)
log.info("Leg 3 complete (-X)")
time.sleep(5)

for i in range(size):
    drone.send_movement_command_XYZ(0,-speed,0)
    time.sleep(0.02)
log.info("Leg 4 complete (-Y)")
time.sleep(5)

# fly rectangle with YAW
log.info("Starting rectangle flight (YAW)...")
for i in range(4):
    if i > 0:
        drone.send_movement_command_YAW(90)
        log.info(f"YAW turn {i} complete")
    for j in range(size):
        drone.send_movement_command_XYZ(speed,0,0)
        time.sleep(0.02)
    log.info(f"YAW leg {i+1} complete")
    time.sleep(5)

log.info("Landing...")
drone.land()
log.info("=== drone_test.py finished ===")
