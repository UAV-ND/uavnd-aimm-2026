#!/usr/bin/env python3

import Jetson.GPIO as GPIO
import time

# Use BOARD pin numbering (physical pins on the header)
GPIO.setmode(GPIO.BOARD)

# Choose the pin (example: pin 12)
PIN = 16

try:
    # Setup pin as output
    GPIO.setup(PIN, GPIO.OUT)

    print(f"Setting pin {PIN} HIGH")

    # Set pin HIGH
    GPIO.output(PIN, GPIO.HIGH)

    # Keep it HIGH for a while
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopping...")

finally:
    GPIO.output(PIN, GPIO.LOW)
    GPIO.cleanup()
