# UAVND AIMM — Autonomous mission (MVP)

Python mission logic for a multirotor running on a companion computer (Jetson-class) with **ArduPilot** over **MAVLink**. The vehicle flies an uploaded AUTO mission until a handoff waypoint, then runs **guided** behavior: spiral search, vision-based centering, payload release, boat GPS transit, landing marker alignment, lidar-assisted descent, and **LAND**.

## Mission state flow

The orchestrator is `mission_mvp.py`. States are executed in order unless the pilot selects **LOITER** (takeover) or a timeout/failure path sends the mission to `manual_override` → `done`.

1. **wait_for_auto_start** — Arm, load payload waypoint from FC, wait for **AUTO**.
2. **wait_for_handoff** — Monitor mission index until handoff waypoint is active.
3. **guided_handoff** — Switch to **GUIDED**, hold position.
4. **search_for_buoy** — Expanding square spiral (3 m step, up to 16 legs), class `black_buoy`.
5. **center_on_buoy** — PID centering, stable confirmation.
6. **hold_at_buoy** — Short hold at the buoy.
7. **search_for_payload** — Spiral from current position, class `target`.
8. **center_payload_target** — PID centering, stable confirmation.
9. **drop_payload** — GPIO pulse to release payload.
10. **wait_for_boat_gps** — Wait for a valid boat fix from MAVLink (see Boat GPS below).
11. **goto_boat_gps** — Fly toward boat; re-command if the fix moves farther than a threshold (default 5 m).
12. **search_boat** — Spiral, class `boat`.
13. **align_over_boat** — PID over landing marker.
14. **descend_on_boat** — Staged descent using downward camera + lidar.
15. **final_land** — **LAND** mode.
16. **manual_override** / **done** — Stop on pilot takeover or failure; clean exit on success.

### Variant: `mission_mvp_rtl.py`

Same steps **1–9** as above, then **return_to_launch** — commands ArduPilot **RTL** (return to home) and waits until the vehicle disarms instead of boat GPS, boat vision, lidar descent, and **LAND** on the deck. Does not start `boat_radio`, lidar, or the `boat`-class detector (only buoy + payload vision and GPIO drop).

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

Boat position on the aircraft is expected as **`BOAT_LAT`** / **`BOAT_LON`** via `NAMED_VALUE_FLOAT` (e.g. FC Lua script `boat_gps.lua` reading a radio string and injecting into MAVLink). See comments in `boat_radio.py`.

## Documentation

- [manuals/NVIDIA Jetson Nano setup manual.md](manuals/NVIDIA%20Jetson%20Nano%20setup%20manual.md)
- [manuals/Linux wireless AP manual.md](manuals/Linux%20wireless%20AP%20manual.md)

## Requirements

- **ArduPilot** + companion link (UDP or serial) as configured in `mission_mvp.py` (`UDP_CONNECTION`).
- **NVIDIA Jetson** (or similar) with TensorRT, OpenCV GStreamer pipeline, and Jetson.GPIO for payload drop.
- **Python**: DroneKit, `simple_pid`, pymavlink; see imports in `modules/drone.py` and `detector_adapter.py`.
- **TensorRT engine** `engine.engine` (classes: `black_buoy`, `target`, `boat`) — see `Tools/` for conversion helpers.
- **Camera** compatible with the GStreamer pipeline in `cv/models/minimal_cv.py`.
- **Lidar** on the serial port set in `mission_mvp.py` (`LIDAR_PORT`).

## Configuration

Important constants live at the top of `mission_mvp.py`:

- Connection strings, altitudes, spiral geometry, timeouts, PID tolerances.
- `HANDOFF_MISSION_INDEX` / `PAYLOAD_TARGET_MISSION_INDEX` — must match the uploaded mission on the FC.
- **Paths**: `sys.path` and `log_path` may still point at a legacy machine path; set them to your checkout (and log directory) before flight.

## Running the mission

From the repository root (so `modules` resolves correctly):

```sh
python3 mission_mvp.py
```

RTL variant (home return after payload drop):

```sh
python3 mission_mvp_rtl.py
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
| `mission_mvp.py` | Main state machine (buoy → payload → boat landing) |
| `mission_mvp_rtl.py` | Same through payload drop, then **RTL** instead of boat landing |
| `modules/` | Flight control (`control.py`, `drone.py`, `lidar.py`) |
| `detector_adapter.py` | Multi-class TensorRT detector + optional GPIO |
| `target_detector_adapter.py` | Legacy single-pipeline detector (`cv_2`); not used by `mission_mvp.py` |
| `boat_radio.py` | Boat GPS from MAVLink |
| `boat_gps.lua` | Example FC-side Lua for injecting boat lat/lon |
| `cv/models/` | Vision inference (`minimal_cv.py` used by MVP; `cv_2.py` alternate) |
| `test/` | Hardware/integration checks |
| `Tools/` | Model conversion, plotting, utilities |
| `manuals/` | Jetson and networking setup notes |

## Safety

This software moves a real aircraft. Verify parameters, mission indices, connection strings, and RC failsafe before flight. Experimental code: use at your own risk; authors are not responsible for injury or damage.
