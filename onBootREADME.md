# UAV-ND AIMM Drone Controller

Two systemd services run on boot: **`mavproxy.service`** connects to the CubeOrange flight controller on `/dev/ttyACM0` and forwards MAVLink to `udp:127.0.0.1:14552`. **`drone_controller.service`** then runs `drone_test.py`, which connects via DroneKit on `udpin:127.0.0.1:14552`. The drone controller will not start until MAVProxy is up.

---

## Setup

```bash
# Install and enable services
sudo systemctl daemon-reload
sudo systemctl enable mavproxy.service drone_controller.service
sudo systemctl start mavproxy.service drone_controller.service

# Optional: passwordless sudo for uav-nano
sudo visudo  # add: uav-nano ALL=(ALL) NOPASSWD: ALL
```

**Key service settings** — both files live in `/etc/systemd/system/`:
- MAVProxy: `WorkingDirectory=/home/uav-nano`, `--logfile /home/uav-nano/mav.tlog`, `--non-interactive`, `ExecStartPre=/bin/sleep 10`
- Drone controller: `After=mavproxy.service`, `Requires=mavproxy.service`, `ExecStartPre=/bin/sleep 15`

---

## Monitoring

```bash
sudo systemctl status mavproxy.service        # quick health check
sudo journalctl -u mavproxy.service -f        # live logs
sudo journalctl -u drone_controller.service -f
sudo journalctl -u mavproxy.service -b        # logs since last boot
```

---

## Common Issues

| Symptom | Fix |
|---|---|
| `Permission denied: mav.tlog` | Ensure `WorkingDirectory=/home/uav-nano` and `--logfile` flag are set |
| DroneKit `Link timeout, no heartbeat` | Check MAVProxy is healthy first; it must be running before the script starts |
| `Waiting for heartbeat` forever | Run `ls /dev/ttyACM*` to confirm FC is detected; test MAVProxy manually |
| Service won't start after edits | Always run `sudo systemctl daemon-reload` after changing service files |

---

## Manual Control

```bash
sudo systemctl restart mavproxy.service
sudo systemctl restart drone_controller.service
sudo systemctl stop mavproxy.service drone_controller.service
python3 .../test/drone_test.py                # run script manually (MAVProxy must be running first)
```
