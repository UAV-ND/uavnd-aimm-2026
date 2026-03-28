import sys, time
sys.path.insert(1,'/home/uav-nano/Documents/aimm-dev/uavnd-aimm-2026/modules')
import drone

# config
height = 1
speed = 3   # m/s
size = 165  # 10 meter

v = drone.connect_drone('udpin:127.0.0.1:14552')
print("Connected:", v is not None)
print("Mode:", v.mode)
print("Armed:", drone.vehicle.armed if hasattr(drone, "vehicle") else "n/a")

# Wait for RC arm signal
print("Waiting for RC arm...")
while not drone.vehicle.armed:
    print("Not armed yet, waiting...")
    time.sleep(1)

print("Armed! Proceeding with flight...")

drone.arm_and_takeoff(height)

print(drone.get_version())
print(drone.get_mode())
print(drone.get_location())
time.sleep(5)

# fly rectangle with XYZ
for i in range(size):
    drone.send_movement_command_XYZ(speed,0,0)
    time.sleep(0.02)
time.sleep(5)
for i in range(size):
    drone.send_movement_command_XYZ(0,speed,0)
    time.sleep(0.02)
time.sleep(5)
for i in range(size):
    drone.send_movement_command_XYZ(-speed,0,0)
    time.sleep(0.02)
time.sleep(5)
for i in range(size):
    drone.send_movement_command_XYZ(0,-speed,0)
    time.sleep(0.02)
time.sleep(5)

# fly rectangle with YAW
for i in range(4):
    if i > 0:
        drone.send_movement_command_YAW(90)
    for i in range(size):
        drone.send_movement_command_XYZ(speed,0,0)
        time.sleep(0.02)
    time.sleep(5)

drone.land()
