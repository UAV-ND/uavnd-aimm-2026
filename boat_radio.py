#!/usr/bin/env python3
import serial
import time

ser = None


def connect_boat_radio(port, baudrate=57600, timeout=1.0):
    global ser
    ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
    return ser


def disconnect_boat_radio():
    global ser
    if ser is not None:
        ser.close()
        ser = None


def read_boat_gps():
    """
    Expected line format:
    BOAT,41.698123,-86.237456
    """
    global ser
    if ser is None:
        raise RuntimeError("Boat radio not connected")

    line = ser.readline().decode("utf-8", errors="ignore").strip()
    if not line:
        return None

    parts = line.split(",")
    if len(parts) != 3:
        return None

    if parts[0] != "BOAT":
        return None

    try:
        lat = float(parts[1])
        lon = float(parts[2])
        return {"lat": lat, "lon": lon, "timestamp": time.time()}
    except Exception:
        return None