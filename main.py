"""
Memento Mare API - FastAPI server
All wave model physics run here. ESP32 just fetches and displays.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import math
import asyncio
from datetime import datetime, timezone, timedelta

app = FastAPI(title="Memento Mare API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ============================================================
# SPOTS
# ============================================================
# Each spot has a list of buoys with their direction window
# dir_min/dir_max = range of swell directions this buoy is authoritative for
# If a swell component's direction falls in this window, use this buoy for it
SPOTS = {
    "washout": {
        "name":"WASHOUT", "lat":32.646, "lon":-79.941, "orientation":100,
        "tide_station":"8666467", "Ks":1.495, "near_depth":4.0, "tz_offset":-4,
        "face_mult":1.0,  # East Coast reports actual Hs
        "buoys": [
            {"id":"41004", "dist_km":81,  "dir_min":0,   "dir_max":360},  # primary — all directions
            {"id":"41025", "dist_km":495, "dir_min":0,   "dir_max":90},   # NE swells
        ]
    },
    "iop": {
        "name":"IOP", "lat":32.787, "lon":-79.771, "orientation":95,
        "tide_station":"8666467", "Ks":1.414, "near_depth":5.0, "tz_offset":-4,
        "face_mult":1.0,
        "buoys": [
            {"id":"41004", "dist_km":70,  "dir_min":0,   "dir_max":360},
            {"id":"41025", "dist_km":473, "dir_min":0,   "dir_max":90},
        ]
    },
    "huntington": {
        "name":"HUNTINGTON", "lat":33.654, "lon":-118.003, "orientation":270,
        "tide_station":"9410660", "Ks":1.495, "near_depth":4.0, "tz_offset":-7,
        "face_mult":1.7,  # California surf reports face height ~1.5-2x Hs
        "buoys": [
            {"id":"46086", "dist_km":141, "dir_min":130, "dir_max":260},  # S/SW swells
            {"id":"46047", "dist_km":197, "dir_min":260, "dir_max":360},  # NW/W swells
            {"id":"46025", "dist_km":98,  "dir_min":260, "dir_max":360},  # NW wind swell backup
        ]
    },
    "blacks": {
        "name":"BLACKS", "lat":32.856, "lon":-117.253, "orientation":270,
        "tide_station":"9410170", "Ks":1.622, "near_depth":6.0, "tz_offset":-7,
        "face_mult":1.7,
        "buoys": [
            {"id":"46086", "dist_km":90,  "dir_min":130, "dir_max":260},  # S/SW swells
            {"id":"46047", "dist_km":218, "dir_min":260, "dir_max":360},  # NW/W swells
        ]
    },
    "pipeline": {
        "name":"PIPELINE", "lat":21.665, "lon":-158.053, "orientation":330,
        "tide_station":"1612340", "Ks":1.778, "near_depth":2.0, "tz_offset":-10,
        "face_mult":0.5,  # Hawaii uses Hawaiian scale (~half of Hs)
        "buoys": [
            {"id":"51001", "dist_km":476, "dir_min":270, "dir_max":360},  # N/NW swells
            {"id":"51201", "dist_km":7,   "dir_min":0,   "dir_max":360},  # local reference
            {"id":"51004", "dist_km":751, "dir_min":130, "dir_max":270},  # south swells
        ]
    },
}

# ============================================================
# WAVE PHYSICS
# ============================================================
def refraction_kr(theta_deg, d_off, d_near):
    if theta_deg > 89: theta_deg = 89
    C_off  = math.sqrt(9.81 * d_off)
    C_near = math.sqrt(9.81 * d_near)
    sin_t  = min(math.sin(math.radians(theta_deg)) * (C_near / C_off), 0.999)
    cos_off  = math.cos(math.radians(theta_deg))
    cos_near = max(math.sqrt(1 - sin_t**2), 0.01)
    return math.sqrt(cos_off / cos_near)

def swell_angle(swell_dir, orientation):
    # orientation = direction beach faces = direction waves come FROM to hit it
    # e.g. Washout faces ESE (100deg) = waves come from 100deg to hit beach
    diff = abs(swell_dir - orientation)
    if diff > 180: diff = 360 - diff
    return float(diff)

def period_factor(p):
    if p >= 14: return 1.15
    if p >= 10: return 1.08
    if p >= 8:  return 1.03
    return 1.0

def nearshore_ft(off_m, period, swell_dir, spot):
    Hs    = off_m * spot["Ks"]
    Kr    = refraction_kr(swell_angle(swell_dir, spot["orientation"]), 20.0, spot["near_depth"])
    H     = Hs * Kr * period_factor(period)
    return round(H * 3.28084, 2)

def wind_label(wd, orientation):
    # Offshore wind blows FROM land TO sea = opposite of beach facing direction
    offshore_dir = (orientation + 180) % 360
    diff = abs(wd - offshore_dir)
    if diff > 180: diff = 360 - diff
    if diff <= 45:  return "OFFSHORE"
    if diff <= 75:  return "SIDE-OFF"
    if diff <= 105: return "SIDE-ON"
    return "ONSHORE"

def wind_quality(wd, orientation, wind_mph=0):
    l = wind_label(wd, orientation)
    if l in ("OFFSHORE","SIDE-OFF"): return "good"
    if l == "SIDE-ON": return "marginal"
    # Onshore — only bad if strong enough to matter
    if wind_mph < 8: return "marginal"  # light onshore is barely noticeable
    return "bad"

def calc_stars(ht, period, swell_dir, wd, wmph, spot):
    if ht < 0.5: return 0
    s = 0
    if ht >= 3.0: s += 2
    elif ht >= 1.5: s += 1
    if wind_label(wd, spot["orientation"]) in ("OFFSHORE","SIDE-OFF") and wmph < 20: s += 1
    if period >= 10: s += 1
    return min(s, 4)

def cond_label(stars, ht):
    if ht < 0.5:   return "FLAT"
    if stars >= 4:  return "EPIC"
    if stars >= 3:  return "GOOD"
    if stars >= 2:  return "FAIR"
    return "POOR"

# ============================================================
# FETCHERS
# ============================================================
async def fetch_with_retry(client, url, attempts=3, timeout=15):
    for i in range(attempts):
        try:
            r = await client.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            print(f"HTTP {r.status_code} for {url[:80]}")
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
        if i < attempts-1:
            await asyncio.sleep(2)
    return None

async def fetch_ndbc(client, buoy_id):
    """
    Parse NDBC .spec file for separate swell components.
    Returns list of swells: [{height_m, period_s, direction_deg}, ...]
    We apply shoaling to each component separately then sum in quadrature.
    """
    try:
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{buoy_id}.spec"
        r = await client.get(url, timeout=15)
        if r.status_code != 200:
            print(f"NDBC {buoy_id} spec: HTTP {r.status_code}, falling back to .txt")
            # Fallback to standard met file
            url2 = f"https://www.ndbc.noaa.gov/data/realtime2/{buoy_id}.txt"
            r = await client.get(url2, timeout=15)
            if r.status_code != 200:
                return None
            lines = [l for l in r.text.strip().split("\n") if l.strip()]
            if len(lines) < 3: return None
            headers = lines[0].lstrip("#").split()
            data    = lines[2].split()
            obs = {headers[i]: data[i] for i in range(min(len(headers), len(data)))}
            def sf(k):
                v = obs.get(k, "MM")
                try:
                    f = float(v)
                    return None if f > 900 else f
                except: return None
            wvht = sf("WVHT")
            dpd  = sf("DPD") or sf("APD")
            mwd  = sf("MWD")
            if wvht is None: return None
            return [{"height_m": wvht, "period_s": dpd or 8.0, "direction_deg": int(mwd or 270)}]

        lines = [l for l in r.text.strip().split("\n") if l.strip()]
        if len(lines) < 3: return None

        # Parse headers — spec file has two header rows
        headers = lines[0].lstrip("#").split()
        data    = lines[2].split()
        obs = {headers[i]: data[i] for i in range(min(len(headers), len(data)))}

        def sf(k):
            v = obs.get(k, "MM")
            try:
                f = float(v)
                return None if f > 900 else f
            except: return None

        swells = []

        # Primary swell
        swh = sf("SwH"); swp = sf("SwP"); swd = sf("SwD")
        if swh and swh > 0.05:
            # SwD is reported as compass direction string (N, NE, etc.) or degrees
            try:
                d = float(swd) if swd else 270
            except:
                dirs = {"N":0,"NNE":22,"NE":45,"ENE":67,"E":90,"ESE":112,"SE":135,
                        "SSE":157,"S":180,"SSW":202,"SW":225,"WSW":247,"W":270,
                        "WNW":292,"NW":315,"NNW":337}
                d = dirs.get(str(swd), 270)
            swells.append({"height_m": swh, "period_s": swp or 10.0, "direction_deg": int(d)})

        # Wind waves
        wwh = sf("WWH"); wwp = sf("WWP"); wwd = sf("WWD")
        if wwh and wwh > 0.05:
            swells.append({"height_m": wwh, "period_s": wwp or 6.0, "direction_deg": int(wwd or 270)})

        print(f"NDBC {buoy_id} spec: {len(swells)} components: {[(s['height_m'],s['period_s'],s['direction_deg']) for s in swells]}")
        return swells if swells else None

    except Exception as e:
        print(f"NDBC {buoy_id} error: {e}")
        return None

async def fetch_wavewatch3(client, spot):
    """
    Fetch NOAA WAVEWATCH III 24hr forecast via Open-Meteo marine forecast
    Uses marine-api.open-meteo.com hourly at index 24 (24hrs from now)
    More reliable than ERDDAP for our use case
    """
    try:
        lat = spot["lat"]
        lon = spot["lon"]
        url = (
            f"https://marine-api.open-meteo.com/v1/marine"
            f"?latitude={lat}&longitude={lon}"
            f"&hourly=wave_height,wave_period,wave_direction"
            f",swell_wave_height,swell_wave_period,swell_wave_direction"
            f",&forecast_days=3&timezone=GMT"
        )
        r = await client.get(url, timeout=15)
        if r.status_code != 200:
            print(f"WW3/marine HTTP {r.status_code}")
            return None
        data = r.json()
        hourly = data.get("hourly", {})
        wh  = hourly.get("wave_height", [])
        wp  = hourly.get("wave_period", [])
        wd  = hourly.get("wave_direction", [])
        swh = hourly.get("swell_wave_height", [])
        swp = hourly.get("swell_wave_period", [])
        swd = hourly.get("swell_wave_direction", [])

        # Index 24 = 24hrs from now
        idx = min(24, len(wh)-1)
        swells = []
        h1 = wh[idx] if wh else None
        p1 = wp[idx] if wp else 8.0
        d1 = wd[idx] if wd else 270
        h2 = swh[idx] if swh else None
        p2 = swp[idx] if swp else 10.0
        d2 = swd[idx] if swd else 270

        if h1 and h1 > 0.05:
            swells.append({"height_m": float(h1), "period_s": float(p1 or 8), "direction_deg": int(d1 or 270)})
        if h2 and h2 > 0.05 and h2 != h1:
            swells.append({"height_m": float(h2), "period_s": float(p2 or 10), "direction_deg": int(d2 or 270)})

        print(f"WW3 {spot['name']} tmrw: {[(s['height_m'],s['period_s'],s['direction_deg']) for s in swells]}")
        return swells if swells else None
    except Exception as e:
        print(f"WW3 error {spot['name']}: {e}")
        return None

async def fetch_marine(client, spot):
    url = (f"https://marine-api.open-meteo.com/v1/marine"
           f"?latitude={spot['lat']}&longitude={spot['lon']}"
           f"&current=wave_height,wave_period,wave_direction"
           f"&hourly=wave_height,wave_period,wave_direction"
           f"&forecast_days=2&timezone=America%2FNew_York")
    return await fetch_with_retry(client, url)

async def fetch_wind(client, spot):
    url = (f"https://api.open-meteo.com/v1/forecast"
           f"?latitude={spot['lat']}&longitude={spot['lon']}"
           f"&current=wind_speed_10m,wind_direction_10m"
           f"&hourly=wind_speed_10m,wind_direction_10m"
           f"&wind_speed_unit=mph&forecast_days=2&timezone=America%2FNew_York")
    data = await fetch_with_retry(client, url)
    if data: return data
    return {"current":{"wind_speed_10m":0,"wind_direction_10m":0},
            "hourly":{"wind_speed_10m":[0]*48,"wind_direction_10m":[0]*48}}

async def fetch_tide_now(client, station):
    url = (f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
           f"?date=latest&station={station}&product=water_level"
           f"&datum=MLLW&time_zone=lst_ldt&units=english&format=json")
    data = await fetch_with_retry(client, url)
    if not data: return 3.0
    pts = data.get("data",[])
    return float(pts[-1]["v"]) if pts else 3.0

async def fetch_tide_hilo(client, station):
    """Get today's hi/lo for min/max range"""
    url = (f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
           f"?date=today&station={station}&product=predictions&interval=hilo"
           f"&datum=MLLW&time_zone=lst_ldt&units=english&format=json")
    data = await fetch_with_retry(client, url)
    if not data: return 0.0, 6.0
    vals = [float(p["v"]) for p in data.get("predictions",[])]
    return (min(vals), max(vals)) if vals else (0.0, 6.0)

async def fetch_tide_curve(client, station, tz_offset=-4):
    """
    Returns tide data for plotting as (time, elevation) graph:
    - window: -1hr to +24hr from now (25hr window)
    - y_min: floor of lowest low tide in window
    - y_max: ceil of highest high tide in window
    - points: list of {x_frac, y_norm} for each hi/lo event in window
      x_frac: 0.0 = left edge (-1hr), 1.0 = right edge (+24hr)
      y_norm: 0.0 = y_min, 1.0 = y_max
    - now_x_frac: where "now" sits on the x axis = 1/25 = 0.04
    - current_y_norm: current tide height normalized
    """
    from datetime import datetime as dt

    now_utc   = datetime.now(timezone.utc)
    now_local = now_utc + timedelta(hours=tz_offset)
    today     = now_local.date()
    now_mins  = now_local.hour * 60 + now_local.minute

    # Window: 18 hours total, now sits 1.5 hours past x=0
    # x=0 = 1.5hrs ago, x=63 = 16.5hrs from now
    win_start = now_mins - 90   # 1.5 hours ago
    win_end   = now_mins + 990  # 16.5 hours from now
    win_span  = 1080            # 18 hours in minutes

    # Fetch today + tomorrow + day after hilo predictions
    today_str    = now_local.strftime("%Y%m%d")
    tomorrow_str = (now_local + timedelta(days=1)).strftime("%Y%m%d")
    day3_str     = (now_local + timedelta(days=2)).strftime("%Y%m%d")
    day4_str     = (now_local + timedelta(days=3)).strftime("%Y%m%d")

    url1 = (f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
            f"?begin_date={today_str}&end_date={tomorrow_str}"
            f"&station={station}&product=predictions&interval=hilo"
            f"&datum=MLLW&time_zone=lst_ldt&units=english&format=json")
    url2 = (f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
            f"?begin_date={tomorrow_str}&end_date={day3_str}"
            f"&station={station}&product=predictions&interval=hilo"
            f"&datum=MLLW&time_zone=lst_ldt&units=english&format=json")
    url3 = (f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
            f"?begin_date={day3_str}&end_date={day4_str}"
            f"&station={station}&product=predictions&interval=hilo"
            f"&datum=MLLW&time_zone=lst_ldt&units=english&format=json")

    data1, data2, data3 = await asyncio.gather(
        fetch_with_retry(client, url1),
        fetch_with_retry(client, url2),
        fetch_with_retry(client, url3),
        return_exceptions=True
    )

    preds = []
    if isinstance(data1, dict): preds += data1.get("predictions", [])
    if isinstance(data2, dict): preds += data2.get("predictions", [])
    if isinstance(data3, dict): preds += data3.get("predictions", [])
    print(f"Tide preds fetched: {len(preds)} total events")
    if len(preds) < 2: return {}

    # Parse all events into minutes-since-midnight-today
    events = []
    for p in preds:
        try:
            t = dt.strptime(p["t"], "%Y-%m-%d %H:%M")
            delta_days = (t.date() - today).days
            mins = delta_days * 1440 + t.hour * 60 + t.minute
            val  = float(p["v"])
            typ  = p.get("type", "H")
            if not any(abs(e[0] - mins) < 5 for e in events):
                events.append((mins, val, typ))
        except:
            pass
    events.sort(key=lambda x: x[0])

    if len(events) < 2: return {}

    # Get events in window PLUS one before and one after for curve continuity
    window_events = []
    before = None
    after  = None
    for i, (m, v, t) in enumerate(events):
        if m < win_start:
            before = (m, v, t)  # keep updating — we want the last one before window
        elif m <= win_end:
            window_events.append((m, v, t))
        elif after is None:
            after = (m, v, t)  # first event after window

    if before:
        window_events.insert(0, before)
    if after:
        window_events.append(after)

    if len(window_events) < 2: return {}

    # Y axis: floor of lowest low, ceil of highest high in window
    window_vals = [v for _, v, _ in window_events]
    # Dynamic y range based on actual tides in window
    # Add 20% padding above and below for visual breathing room
    raw_min = min(window_vals)
    raw_max = max(window_vals)
    padding = max((raw_max - raw_min) * 0.2, 0.3)
    y_min = raw_min - padding
    y_max = raw_max + padding
    y_range = y_max - y_min

    # Current tide via cosine interpolation
    current_val = y_min + (y_max - y_min) / 2  # fallback
    for i in range(len(events)-1):
        m0, v0, _ = events[i]
        m1, v1, _ = events[i+1]
        if m0 <= now_mins <= m1:
            span = m1 - m0
            frac = (now_mins - m0) / span if span > 0 else 0
            current_val = v0 + (v1 - v0) * (1 - math.cos(frac * math.pi)) / 2
            break

    # Pre-compute tide height for each of 64 pixels
    # Each pixel = win_span/64 minutes
    mins_per_px = win_span / 64
    curve = []
    for px in range(64):
        t_mins = win_start + px * mins_per_px  # absolute minutes since midnight

        # Find surrounding hi/lo events and cosine interpolate
        val = current_val  # fallback
        for i in range(len(events)-1):
            m0, v0, _ = events[i]
            m1, v1, _ = events[i+1]
            if m0 <= t_mins <= m1:
                span = m1 - m0
                frac = (t_mins - m0) / span if span > 0 else 0
                val = v0 + (v1 - v0) * (1 - math.cos(frac * math.pi)) / 2
                break
        else:
            # Outside all events — use nearest endpoint
            if t_mins < events[0][0]:
                val = events[0][1]
            else:
                val = events[-1][1]

        y_norm = max(0.0, min(1.0, (val - y_min) / y_range))
        curve.append(round(y_norm, 4))

    now_x_frac     = 90 / win_span  # always 0.0833
    current_y_norm = max(0.0, min(1.0, (current_val - y_min) / y_range))
    now_px         = round(now_x_frac * 63)  # pixel index of now (~5)

    print(f"Tide {station}: y={y_min}-{y_max}ft now_px={now_px} curve[{now_px}]={curve[now_px]:.3f}")
    return {
        "curve":      curve,       # 64 pre-computed normalized values
        "now_px":     now_px,      # pixel index of now
        "current_y":  round(current_y_norm, 4),
        "y_min":      y_min,
        "y_max":      y_max,
    }

async def fetch_water_temp(client, spot):
    url = (f"https://marine-api.open-meteo.com/v1/marine"
           f"?latitude={spot['lat']}&longitude={spot['lon']}"
           f"&current=sea_surface_temperature&temperature_unit=fahrenheit")
    data = await fetch_with_retry(client, url)
    if data: return data["current"].get("sea_surface_temperature", 0)
    return 0

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def root():
    return {"service":"Memento Mare API","spots":list(SPOTS.keys()),"usage":"/surf/{spot_id}"}

@app.get("/spots")
def get_spots():
    return [{"id":k,"name":v["name"]} for k,v in SPOTS.items()]

@app.get("/surf/{spot_id}")
async def get_surf(spot_id: str):
    spot = SPOTS.get(spot_id.lower())
    if not spot:
        raise HTTPException(404, f"Spot '{spot_id}' not found")


    async with httpx.AsyncClient(verify=False) as client:
        spot_buoys = spot.get("buoys", [])
        buoy_ids   = list({b["id"] for b in spot_buoys})

        tasks = [
            fetch_marine(client, spot),
            fetch_wind(client, spot),
            fetch_tide_now(client, spot["tide_station"]),
            fetch_tide_hilo(client, spot["tide_station"]),
            fetch_tide_curve(client, spot["tide_station"], spot.get("tz_offset", -4)),
            fetch_water_temp(client, spot),
            fetch_wavewatch3(client, spot),
        ] + [fetch_ndbc(client, bid) for bid in buoy_ids]

        results = await asyncio.gather(*tasks, return_exceptions=True)

    marine     = results[0]
    wind       = results[1]
    tide_now   = results[2]
    tide_hilo  = results[3]
    tide_curve = results[4]
    water_temp = results[5]
    ww3_tmrw   = results[6] if isinstance(results[6], list) else None
    buoy_data  = {}
    for i, bid in enumerate(buoy_ids):
        r = results[7+i]
        if isinstance(r, list) and r:
            buoy_data[bid] = r


    # Multi-buoy wave height calculation
    # Each buoy contributes swells within its directional window
    # Sum all components in quadrature: H_total = sqrt(sum(H_i^2))
    energy_sum = 0.0
    best_p = 8.0
    best_d = 90
    best_e = 0.0
    used_buoys = []

    if buoy_data:
        for buoy_cfg in spot.get("buoys", []):
            bid = buoy_cfg["id"]
            swells = buoy_data.get(bid, [])
            dir_min = buoy_cfg["dir_min"]
            dir_max = buoy_cfg["dir_max"]

            for s in swells:
                d = s["direction_deg"]
                # Check if swell direction falls in this buoy's window
                in_window = False
                if dir_min <= dir_max:
                    in_window = dir_min <= d <= dir_max
                else:  # wraps around 360
                    in_window = d >= dir_min or d <= dir_max

                if in_window:
                    h_near = nearshore_ft(s["height_m"], s["period_s"], d, spot)
                    energy_sum += h_near ** 2
                    if h_near > best_e:
                        best_e = h_near
                        best_p = s["period_s"]
                        best_d = d
                    used_buoys.append(f"{bid}:{s['height_m']}m@{d}°->{h_near:.1f}ft")

        if energy_sum > 0:
            ht_ft = round(math.sqrt(energy_sum), 2)
            print(f"Multi-buoy {spot['name']}: {used_buoys} -> {ht_ft}ft")
        else:
            # No buoy data matched — fall back to Open-Meteo
            print(f"No buoy match for {spot['name']}, using Open-Meteo")
            buoy_data = {}

    if not buoy_data or energy_sum == 0:
        if isinstance(marine, dict):
            curr  = marine.get("current", {})
            off_h_m = curr.get("wave_height") or 0.0
            off_p   = curr.get("wave_period") or 6.0
            off_d   = int(curr.get("wave_direction") or 90)
            ht_ft   = nearshore_ft(off_h_m, off_p, off_d, spot)
            best_p  = off_p
            best_d  = off_d
            print(f"Open-Meteo {spot['name']}: {off_h_m}m -> {ht_ft}ft")
        else:
            raise HTTPException(502, "No wave data available")

    off_p = best_p
    off_d = best_d

    # Wind
    wind_data = wind if isinstance(wind, dict) else {"current":{"wind_speed_10m":0,"wind_direction_10m":0},"hourly":{"wind_speed_10m":[0]*48,"wind_direction_10m":[0]*48}}
    wind_mph  = wind_data["current"].get("wind_speed_10m") or 0
    wind_dir  = int(wind_data["current"].get("wind_direction_10m") or 0)
    ws  = wind_data["hourly"]["wind_speed_10m"]
    wdh = wind_data["hourly"]["wind_direction_10m"]
    idx2 = min(33, len(ws)-1)
    tmrw_wmph = ws[idx2] or 0
    tmrw_wdir = int(wdh[idx2] or 0)

    # Tide
    tide_cur = tide_now if isinstance(tide_now, float) else 3.0
    tide_mn, tide_mx = tide_hilo if isinstance(tide_hilo, tuple) else (0.0, 6.0)
    tide_data      = tide_curve if isinstance(tide_curve, dict) else {}
    tide_curve_arr = tide_data.get("curve", [])
    tide_now_px    = tide_data.get("now_px", 5)
    tide_current_y = tide_data.get("current_y", 0.5)
    tide_y_min     = tide_data.get("y_min", -1)
    tide_y_max     = tide_data.get("y_max", 7)
    wtemp = water_temp if isinstance(water_temp, (int,float)) else 0

    # Tomorrow wave height
    # WW3 uses same multi-swell approach as current
    if ww3_tmrw and isinstance(ww3_tmrw, list):
        tmrw_energy = 0.0
        for s in ww3_tmrw:
            h = nearshore_ft(s["height_m"], s["period_s"], s["direction_deg"], spot)
            tmrw_energy += h ** 2
        tmrw_ft = round(math.sqrt(tmrw_energy), 2) if tmrw_energy > 0 else 0.0
        print(f"WW3 tomorrow {spot['name']}: {tmrw_ft}ft")
    elif isinstance(marine, dict) and "hourly" in marine:
        wh = marine["hourly"]["wave_height"]
        wp = marine["hourly"]["wave_period"]
        wd = marine["hourly"]["wave_direction"]
        idx = min(24, len(wh)-1)
        tmrw_ft = nearshore_ft(wh[idx] or 0.0, wp[idx] or 6.0, int(wd[idx] or off_d), spot)
    else:
        tmrw_ft = ht_ft  # fallback to current

    fm      = spot.get("face_mult", 1.0)
    stars   = calc_stars(ht_ft, off_p, off_d, wind_dir, wind_mph, spot)
    cond    = cond_label(stars, ht_ft)
    wlabel  = wind_label(wind_dir, spot["orientation"])
    wqual   = wind_quality(wind_dir, spot["orientation"], wind_mph)

    disp_lo  = round(ht_ft * fm * 0.8,  1) if ht_ft  >= 0.5 else 0.0
    disp_hi  = round(ht_ft * fm * 1.25, 1) if ht_ft  >= 0.5 else 1.0
    tmrw_lo  = round(tmrw_ft * fm * 0.8,  1) if tmrw_ft >= 0.5 else 0.0
    tmrw_hi  = round(tmrw_ft * fm * 1.25, 1) if tmrw_ft >= 0.5 else 1.0
    tmrw_disp = tmrw_ft * fm

    return {
        "spot":         spot["name"],
        "ht":           ht_ft,
        "ht_lo":        disp_lo,
        "ht_hi":        disp_hi,
        "period":       off_p,
        "swell_dir":    off_d,
        "condition":    cond,
        "stars":        stars,
        "wind_mph":     round(float(wind_mph), 1),
        "wind_dir":     wind_dir,
        "wind_label":   wlabel,
        "wind_quality": wqual,
        "tmrw_ht":      tmrw_disp,
        "tmrw_ht_lo":   tmrw_lo,
        "tmrw_ht_hi":   tmrw_hi,
        "tmrw_flat":    tmrw_disp < 0.5,
        "tide_now":     round(tide_cur, 2),
        "tide_min":     round(tide_mn, 2),
        "tide_max":     round(tide_mx, 2),
        "tide_curve":    tide_curve_arr,
        "tide_now_px":   tide_now_px,
        "tide_current_y": tide_current_y,
        "tide_y_min":    tide_y_min,
        "tide_y_max":    tide_y_max,
        "water_temp":   round(float(wtemp), 1),
        "updated":      datetime.utcnow().isoformat() + "Z",
    }
