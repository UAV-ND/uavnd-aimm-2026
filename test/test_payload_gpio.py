#!/usr/bin/env python3
"""
Component test: payload release GPIO (same pin as detector_adapter: BOARD pin 16)

Pulses HIGH for 0.5 s then LOW, then exits — matches mission drop behavior
without loading TensorRT or camera.

Usage (Jetson, often requires sudo for GPIO):
  sudo python3 test/test_payload_gpio.py
  sudo python3 test/test_payload_gpio.py --pulse 0.5
"""
import argparse
import sys
import time

try:
    import Jetson.GPIO as GPIO
except ImportError as e:
    print("ERROR: Jetson.GPIO not available:", e)
    sys.exit(1)


PAYLOAD_PIN = 16


def main():
    parser = argparse.ArgumentParser(description="Pulse payload GPIO once")
    parser.add_argument(
        "--pulse",
        type=float,
        default=0.5,
        help="Seconds to hold pin HIGH",
    )
    parser.add_argument(
        "--pin",
        type=int,
        default=PAYLOAD_PIN,
        help="BOARD pin number",
    )
    args = parser.parse_args()

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(args.pin, GPIO.OUT)
    GPIO.output(args.pin, GPIO.LOW)
    try:
        print("Pin %s HIGH for %.2fs" % (args.pin, args.pulse))
        GPIO.output(args.pin, GPIO.HIGH)
        time.sleep(args.pulse)
    finally:
        GPIO.output(args.pin, GPIO.LOW)
        GPIO.cleanup()
        print("Pin LOW, GPIO cleaned up.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
