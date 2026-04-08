# UAVND AIMM — Autonomous mission (MVP)

Python mission logic for a multirotor running on a companion computer (Jetson-class) with **ArduPilot** over **MAVLink**. On the vehicle, **`drone_controller.service`** runs **`main_controller.py`**: AUTO through handoff, then **guided** buoy and payload vision, payload drop, and **RTL** (return to home). The full deck-landing stack (boat GPS, boat vision, lidar, **LAND**) is in **`mission_mvp.py`**.

## Mission state flow

### `main_controller.py` (Jetson boot — RTL)

Entry point for **`drone_controller.service`**. Does not use `boat_radio`, lidar, or the `boat`-class detector. Unless the pilot takes over (**LOITER** / leaving **AUTO**) or a step times out, states run in order and end at **`done`**.

1. **wait_for_auto_start** — Arm, load payload waypoint from FC, wait for **AUTO**.
2. **wait_for_handoff** — Monitor mission index until handoff waypoint is active.
3. **guided_handoff** — Switch to **GUIDED**, hold position.
4. **search_for_buoy** — Expanding square spiral (3 m step, up to 16 legs), class `black_buoy`.
5. **center_on_buoy** — PID centering, stable confirmation.
6. **hold_at_buoy** — Short hold at the buoy.
7. **search_for_payload** — Spiral from current position, class `target`.
8. **center_payload_target** — PID centering, stable confirmation.
9. **drop_payload** — GPIO pulse to release payload.
10. **return_to_launch** — Command **RTL**, wait until disarm (or timeout).
11. **manual_override** / **done** — Stop on pilot takeover or failure; clean exit on success.

### `mission_mvp.py` (full MVP — boat landing)

States **1–9** match `main_controller.py`, then continues with boat-side guidance and landing:

10. **wait_for_boat_gps** — Wait for a valid boat fix from MAVLink (see Boat GPS below).
11. **goto_boat_gps** — Fly toward boat; re-command if the fix moves farther than a threshold (default 5 m).
12. **search_boat** — Spiral, class `boat`.
13. **align_over_boat** — PID over landing marker.
14. **descend_on_boat** — Staged descent using downward camera + lidar.
15. **final_land** — **LAND** mode.
16. **manual_override** / **done** — Stop on pilot takeover or failure; clean exit on success.

### `mission_mvp_rtl.py`

Standalone script with the same RTL state sequence as **`main_controller.py`**. Use either for bench runs; the vehicle typically runs **`main_controller.py`** via systemd.

## Architecture

```
mission_mvp.py
    ├── modules/control.py   … PID, modes, goto, hold, body velocity
    ├── modules/drone.py     … DroneKit vehicle, MAVLink helpers
    ├── modules/lidar.py     … Serial rangefinder (descent)
    ├── detector_adapter.py  … TensorRT + camera, class `black_buoy` | `target` | `boat`
    │       └── cv/models/minimal_cv.py
    └── boat_radio.py        … NAMED_VALUE_FLOAT listener (boat lat/lon)
```

Boat position on the aircraft is expected as **`BOAT_LAT`** / **`BOAT_LON`** via `NAMED_VALUE_FLOAT` (e.g. FC Lua script `boat_gps.lua` reading a radio string and injecting into MAVLink). See comments in `boat_radio.py`. **`main_controller.py`** does not use `boat_radio.py`.

For **`main_controller.py`** / **`mission_mvp_rtl.py`**, vision uses classes **`black_buoy`** and **`target`** only (no `boat` class or lidar in the loop).

## Documentation

- [manuals/NVIDIA Jetson Nano setup manual.md](manuals/NVIDIA%20Jetson%20Nano%20setup%20manual.md)
- [manuals/Linux wireless AP manual.md](manuals/Linux%20wireless%20AP%20manual.md)

## Requirements

- **ArduPilot** + companion link (UDP or serial) as configured in `main_controller.py` / `mission_mvp.py` (`UDP_CONNECTION`; Jetson MAVProxy uses `udpin:127.0.0.1:14552`).
- **NVIDIA Jetson** (or similar) with TensorRT, OpenCV GStreamer pipeline, and Jetson.GPIO for payload drop.
- **Python**: DroneKit, `simple_pid`, pymavlink; see imports in `modules/drone.py` and `detector_adapter.py`.
- **TensorRT engine** `engine.engine` — full boat mission needs classes `black_buoy`, `target`, `boat`; **`main_controller.py`** only needs **`black_buoy`** and **`target`**. See `Tools/` for conversion helpers.
- **Camera** compatible with the GStreamer pipeline in `cv/models/minimal_cv.py`.
- **Lidar** on the serial port set in `mission_mvp.py` (`LIDAR_PORT`) — not used by **`main_controller.py`**.

## Configuration

Important constants live at the top of **`main_controller.py`** (vehicle) and **`mission_mvp.py`** (full mission):

- Connection strings, altitudes, spiral geometry, timeouts, PID tolerances.
- `HANDOFF_MISSION_INDEX` / `PAYLOAD_TARGET_MISSION_INDEX` — must match the uploaded mission on the FC.
- **Paths**: `sys.path`, `log_path`, and **`ENGINE_PATH`** (`engine.engine` is relative to the process working directory). Set checkout paths to your machine before flight; see **UAV Boot & Logging** for **`WorkingDirectory`** on systemd.

## Running the mission

From the repository root (so `modules` resolves correctly):

```sh
python3 mission_mvp.py
```

RTL variant (home return after payload drop):

```sh
python3 mission_mvp_rtl.py
```

Same RTL mission as systemd runs on the Jetson:

```sh
python3 main_controller.py
```

### Component tests (`test/`)

Run from the **repository root**. Shared import paths are set in `test/_paths.py`.

| Script | What it exercises |
|--------|---------------------|
| `test/test_drone_link.py` | `modules/drone.py` — telemetry only (no motion) |
| `test/test_control_link.py` | `modules/control.py` — connect, PID setup, mode / next WP |
| `test/test_lidar_component.py` | `modules/lidar.py` — serial distance reads |
| `test/test_boat_radio_component.py` | `boat_radio.py` — `BOAT_LAT` / `BOAT_LON` listener |
| `test/test_detector_component.py` | `detector_adapter.GenericDetector` — camera + TensorRT (Jetson) |
| `test/test_payload_gpio.py` | Payload GPIO pulse (BOARD pin 16; often `sudo`) |

Example:

```sh
python3 test/test_drone_link.py --connection udpin:127.0.0.1:14552
python3 test/test_control_link.py
python3 test/test_lidar_component.py --port /dev/ttyTHS1 --count 30
```

Older ad-hoc scripts (`drone_test.py`, `control_test.py`, etc.) may use stale paths; prefer the `test_*_component.py` helpers above.

## Repository layout

| Path | Purpose |
|------|---------|
| `main_controller.py` | RTL mission (buoy → payload → **RTL**); **`drone_controller.service`** entrypoint |
| `mission_mvp.py` | Main state machine (buoy → payload → boat landing) |
| `mission_mvp_rtl.py` | Same RTL sequence as `main_controller.py`; standalone bench script |
| `modules/` | Flight control (`control.py`, `drone.py`, `lidar.py`) |
| `detector_adapter.py` | Multi-class TensorRT detector + optional GPIO |
| `target_detector_adapter.py` | Legacy single-pipeline detector (`cv_2`); not used by `mission_mvp.py` / `main_controller.py` |
| `boat_radio.py` | Boat GPS from MAVLink |
| `boat_gps.lua` | Example FC-side Lua for injecting boat lat/lon |
| `cv/models/` | Vision inference (`minimal_cv.py` used by MVP; `cv_2.py` alternate) |
| `test/` | Hardware/integration checks |
| `Tools/` | Model conversion, plotting, utilities |
| `manuals/` | Jetson and networking setup notes |

## UAV Boot & Logging Guide

This section covers the two systemd services that run on boot on the Jetson companion, where their logs live, and how to check them. Paths like `~/Documents/aimm-dev/uavnd-aimm-2026/` refer to the vehicle checkout; change them if your clone lives elsewhere.

### Services

#### `mavproxy.service`

Starts MAVProxy — bridges the flight controller (`/dev/ttyACM0`) to a UDP port the drone script connects to.

- Waits 10 seconds after boot before starting
- Streams FC telemetry to `udp:127.0.0.1:14552`
- Also writes a binary telemetry log to `~/mav.tlog`

#### `drone_controller.service`

Runs `main_controller.py` — connects to MAVProxy over `udpin:127.0.0.1:14552`, then runs the RTL mission state machine.

- Waits 15 seconds after boot (mavproxy must be up first)
- Requires `mavproxy.service` — will not start if mavproxy fails
- Waits for RC arm via `control.wait_until_armed()` (~1 s loop); arm-wait lines are **`print()`** to stdout — use **`journalctl -u drone_controller.service`** for those. Structured mission lines go to **`main_controller.log`** via `logging`.

### Log file locations

| Log | Path | Written by |
|-----|------|------------|
| Drone script | `main_controller.log` | Python `logging` module — appends across restarts |
| MAVProxy output | `~/logs/mavproxy.log` | bash `>>` redirect in service file — appends across restarts |
| MAVProxy telemetry | `~/mav.tlog` | MAVProxy binary log |

### Cheat sheet

#### Watch logs live

```bash
# Drone script (timestamped — shows arm-wait, flight legs, errors)
tail -f ~/Documents/aimm-dev/uavnd-aimm-2026/main_controller.log

# MAVProxy (FC connection, battery, RC status)
tail -f ~/logs/mavproxy.log
```

#### Review after a flight

```bash
# Full drone log — each boot run separated by === started === lines
cat ~/Documents/aimm-dev/uavnd-aimm-2026/main_controller.log

# Last 50 lines only
tail -n 50 ~/Documents/aimm-dev/uavnd-aimm-2026/main_controller.log

# Quick scan — did it start, arm, any errors?
sudo journalctl -u drone_controller.service -b | grep -E "armed|Armed|ERROR|started"
```

#### journalctl (catches crashes before log files open)

```bash
# Live follow
sudo journalctl -u drone_controller.service -f
sudo journalctl -u mavproxy.service -f

# Current boot only
sudo journalctl -u drone_controller.service -b

# Previous boot
sudo journalctl -u drone_controller.service -b -1
```

#### Service files

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

# Recommended under [Service] for drone_controller.service so relative ENGINE_PATH=engine.engine resolves:
#   WorkingDirectory=/home/uav-nano/Documents/aimm-dev/uavnd-aimm-2026

# Check status
sudo systemctl status mavproxy.service
sudo systemctl status drone_controller.service
```

### Notes

- **systemd 237** is installed on this Jetson Nano. `StandardOutput=append:` requires 240+, so MAVProxy uses a `bash -c '... >> logfile 2>&1'` redirect instead. Do not remove the `/bin/bash -c '...'` wrapper from `mavproxy.service` or logging will break.
- The drone script log is written by Python's `logging` module directly — no systemd redirect needed for `drone_controller.service`.
- **`ENGINE_PATH`** in `main_controller.py` is relative (`engine.engine`). Set **`WorkingDirectory=`** in `drone_controller.service` to your repo root (see cheat sheet above), or the TensorRT engine may not load on boot.
- Both log files use append mode, so multiple boot sessions are preserved in one file. Look for **`=== main_controller.py started ===`** / **`=== main_controller.py finished ===`** to find each session in **`main_controller.log`**.

## Safety

This software moves a real aircraft. Verify parameters, mission indices, connection strings, and RC failsafe before flight. Experimental code: use at your own risk; authors are not responsible for injury or damage.
