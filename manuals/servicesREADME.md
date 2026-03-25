# UAV Boot & Logging Guide

This document covers the two systemd services that run on boot, where their logs live, and how to check them.

---

## Services

### `mavproxy.service`
Starts MAVProxy — bridges the flight controller (`/dev/ttyACM0`) to a UDP port the drone script connects to.
- Waits 10 seconds after boot before starting
- Streams FC telemetry to `udp:127.0.0.1:14552`
- Also writes a binary telemetry log to `~/mav.tlog`

### `drone_controller.service`
Runs `test/drone_test.py` — connects to MAVProxy, waits for RC arm, then executes the flight routine.
- Waits 15 seconds after boot (mavproxy must be up first)
- Requires `mavproxy.service` — will not start if mavproxy fails
- Polls `vehicle.armed` every second and logs each check until armed

---

## Log file locations

| Log | Path | Written by |
|-----|------|------------|
| Drone script | `test/drone_test.log` | Python `logging` module — appends across restarts |
| MAVProxy output | `~/logs/mavproxy.log` | bash `>>` redirect in service file — appends across restarts |
| MAVProxy telemetry | `~/mav.tlog` | MAVProxy binary log |

---

## Cheat sheet

### Watch logs live
```bash
# Drone script (timestamped — shows arm-wait, flight legs, errors)
tail -f ~/Documents/aimm-dev/uavnd-aimm-2026/test/drone_test.log

# MAVProxy (FC connection, battery, RC status)
tail -f ~/logs/mavproxy.log
```

### Review after a flight
```bash
# Full drone log — each boot run separated by === started === lines
cat ~/Documents/aimm-dev/uavnd-aimm-2026/test/drone_test.log

# Last 50 lines only
tail -n 50 ~/Documents/aimm-dev/uavnd-aimm-2026/test/drone_test.log

# Quick scan — did it start, arm, any errors?
sudo journalctl -u drone_controller.service -b | grep -E "armed|Armed|ERROR|started"
```

### journalctl (catches crashes before log files open)
```bash
# Live follow
sudo journalctl -u drone_controller.service -f
sudo journalctl -u mavproxy.service -f

# Current boot only
sudo journalctl -u drone_controller.service -b

# Previous boot
sudo journalctl -u drone_controller.service -b -1
```

### Service files
```bash
# View
cat /etc/systemd/system/drone_controller.service
cat /etc/systemd/system/mavproxy.service

# Edit
sudo nano /etc/systemd/system/drone_controller.service
sudo nano /etc/systemd/system/mavproxy.service

# After any edit — always reload then restart
sudo systemctl daemon-reload
sudo systemctl restart mavproxy.service drone_controller.service

# Check status
sudo systemctl status mavproxy.service
sudo systemctl status drone_controller.service
```

---

## Notes

- **systemd 237** is installed on this Jetson Nano. `StandardOutput=append:` requires 240+, so MAVProxy uses a `bash -c '... >> logfile 2>&1'` redirect instead. Do not remove the `/bin/bash -c '...'` wrapper from `mavproxy.service` or logging will break.
- The drone script log is written by Python's `logging` module directly — no systemd redirect needed for `drone_controller.service`.
- Both log files use append mode, so multiple boot sessions are preserved in one file. Look for `=== drone_test.py started ===` to find the start of each session in the drone log.
