import sys, time
sys.path.insert(1, '/home/uav-nano/Documents/aimm-dev/Autonomous-Ai-drone-scripts/modules')

import drone

#debug
#print("Mode:", drone.get_mode())
#print("Version:", drone.get_version())

#config 
height = 1
speed = 3 #m/s
size = 165 #10 meter  
#end config

#drone.connect_drone('/dev/ttyACM0')
#drone.connect_drone('udp127.0.0.1:14550')
#drone.connect_drone('udp:127.0.0.1:14552')
#drone.connect_drone('udpin:127.0.0.1:14552')
v = drone.connect_drone('udpin:127.0.0.1:14552')
print("Connected:", v is not None)
print("Mode:", v.mode)

print("Mode:", drone.vehicle.mode if hasattr(drone, "vehicle") else "n/a")
print("Armed:", drone.vehicle.armed if hasattr(drone, "vehicle") else "n/a")
drone.arm_and_takeoff(height)
#drone.connect_drone('udp:127.0.0.1:14551')

print(drone.get_version())
print(drone.get_mode())
print(drone.get_location())

time.sleep(5)

#fly recangle with XYZ
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

#fly rectangle with YAW 

for i in range(4):
    if i > 0:
        drone.send_movement_command_YAW(90)

    for i in range(size):
        drone.send_movement_command_XYZ(speed,0,0)
        time.sleep(0.02)

    time.sleep(5)

drone.land()
