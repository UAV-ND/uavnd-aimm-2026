-- boat_gps.lua
-- Runs on ArduPilot (Copter 4.3+, requires SERIAL_PASS or scripting serial)
--
-- Reads plain ASCII from TELEM1 in the format:
--   BOAT,41.698123,-86.237456
--
-- Injects two NAMED_VALUE_FLOAT MAVLink messages per fix:
--   name="BOAT_LAT"  value=lat  (degrees, float)
--   name="BOAT_LON"  value=lon  (degrees, float)
--
-- ArduPilot parameter setup (do this in Mission Planner before running):
--   SERIAL1_PROTOCOL = 28   (scripting)
--   SERIAL1_BAUD     = 57   (57600)
--   SCR_ENABLE       = 1
--
-- Place this file in the APM/scripts/ folder on the SD card.

local BAUD      = 57600
local uart      = serial:find_serial(0)   -- scripting serial 0 = SERIAL1 (TELEM1)
local buf       = ""
local send_rate = 100   -- ms between loop ticks

if not uart then
    gcs:send_text(6, "boat_gps.lua: serial not found")
    return
end

uart:begin(BAUD)
uart:set_flow_control(0)
gcs:send_text(6, "boat_gps.lua: started on TELEM1 at 57600")


-- Send a NAMED_VALUE_FLOAT via MAVLink
local function send_named_float(name, value)
    -- name must be <= 10 chars; pad/truncate to fit MAVLink field
    local msg = mavlink:new_named_float(name, value)
    if msg then
        mavlink:send_to_gcs(msg)
    end
end


-- Parse a validated BOAT line and inject MAVLink messages
local function handle_line(line)
    -- Expected: BOAT,<lat>,<lon>
    local prefix, lat_s, lon_s = line:match("^(BOAT),([%-]?%d+%.%d+),([%-]?%d+%.%d+)$")

    if not prefix then
        gcs:send_text(6, "boat_gps: bad line: " .. line)
        return
    end

    local lat = tonumber(lat_s)
    local lon = tonumber(lon_s)

    if not lat or not lon then
        gcs:send_text(6, "boat_gps: parse error")
        return
    end

    -- Sanity check — reject obviously invalid coordinates
    if lat < -90 or lat > 90 or lon < -180 or lon > 180 then
        gcs:send_text(6, "boat_gps: coords out of range")
        return
    end

    send_named_float("BOAT_LAT", lat)
    send_named_float("BOAT_LON", lon)

    gcs:send_text(6, string.format("boat_gps: lat=%.6f lon=%.6f", lat, lon))
end


-- Main loop
local function update()
    -- Drain all available bytes into buffer
    local n = uart:available()
    if n and n > 0 then
        -- Cap read per tick to avoid stalling the scheduler
        n = math.min(n, 128)
        for _ = 1, n do
            local byte = uart:read()
            if byte and byte >= 0 then
                local ch = string.char(byte)
                if ch == "\n" then
                    -- Strip carriage return if present
                    local line = buf:gsub("\r", ""):match("^%s*(.-)%s*$")
                    if #line > 0 then
                        handle_line(line)
                    end
                    buf = ""
                else
                    buf = buf .. ch
                    -- Safety: prevent unbounded growth if no newline arrives
                    if #buf > 128 then
                        buf = ""
                    end
                end
            end
        end
    end

    return update, send_rate
end

return update, send_rate