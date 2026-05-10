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
SPOTS = {
    "washout":    {"name":"WASHOUT",    "lat":32.646, "lon":-79.941,  "orientation":100, "tide_station":"8665530", "Ks":1.495, "near_depth":4.0, "ndbc_buoy":None},
    "iop":        {"name":"IOP",        "lat":32.787, "lon":-79.771,  "orientation":95,  "tide_station":"8665530", "Ks":1.414, "near_depth":5.0, "ndbc_buoy":None},
    "huntington": {"name":"HUNTINGTON", "lat":33.654, "lon":-118.003, "orientation":270, "tide_station":"9410660", "Ks":1.495, "near_depth":4.0, "ndbc_buoy":"46222"},
    "blacks":     {"name":"BLACKS",     "lat":32.856, "lon":-117.253, "orientation":270, "tide_station":"9410170", "Ks":1.622, "near_depth":6.0, "ndbc_buoy":"46225"},
    "pipeline":   {"name":"PIPELINE",   "lat":21.665, "lon":-158.053, "orientation":330, "tide_station":"1612340", "Ks":1.778, "near_depth":2.0, "ndbc_buoy":"51201"},
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
    shore_normal = (orientation + 180) % 360
    diff = abs(swell_dir - shore_normal)
    return float(360 - diff if diff > 180 else diff)

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
    diff = abs(wd - (orientation + 180) % 360)
    if diff > 180: diff = 360 - diff
    if diff <= 45:  return "OFFSHORE"
    if diff <= 75:  return "SIDE-OFF"
    if diff <= 105: return "SIDE-ON"
    return "ONSHORE"

def wind_quality(wd, orientation):
    l = wind_label(wd, orientation)
    if l in ("OFFSHORE","SIDE-OFF"): return "good"
    if l == "SIDE-ON": return "marginal"
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
    """Parse NDBC standard met text file"""
    try:
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{buoy_id}.txt"
        r = await client.get(url, timeout=15)
        if r.status_code != 200:
            print(f"NDBC {buoy_id}: HTTP {r.status_code}")
            return None
        lines = [l for l in r.text.strip().split("\n") if l.strip()]
        if len(lines) < 3:
            print(f"NDBC {buoy_id}: not enough lines")
            return None
        headers = lines[0].lstrip("#").split()
        data    = lines[2].split()
        obs = {headers[i]: data[i] for i in range(min(len(headers), len(data)))}
        def safe(k):
            v = obs.get(k,"MM")
            try:
                f = float(v)
                return None if f > 900 else f
            except: return None
        wvht = safe("WVHT")
        dpd  = safe("DPD") or safe("APD")
        mwd  = safe("MWD")
        print(f"NDBC {buoy_id}: WVHT={wvht} DPD={dpd} MWD={mwd}")
        if wvht is None: return None
        return {"wave_height": wvht, "wave_period": dpd or 8.0, "wave_direction": int(mwd or 270)}
    except Exception as e:
        print(f"NDBC {buoy_id} error: {e}")
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

async def fetch_tide_curve(client, station):
    """
    Build 24-point tide curve from NOW using real hi/lo predictions.
    - Fetches current water level (the actual starting point)
    - Fetches next 4 hi/lo events
    - Cosine interpolates between them
    - Returns 24 normalized values (0=day low, 1=day high)
    """
    from datetime import datetime as dt

    # Fetch 2 days of hilo predictions to ensure we have enough future events
    url = (f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
           f"?date=today&station={station}&product=predictions&interval=hilo"
           f"&datum=MLLW&time_zone=lst_ldt&units=english&format=json")
    data = await fetch_with_retry(client, url)

    # Also fetch tomorrow to have enough events
    url2 = (f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
            f"?date=tomorrow&station={station}&product=predictions&interval=hilo"
            f"&datum=MLLW&time_zone=lst_ldt&units=english&format=json")
    data2 = await fetch_with_retry(client, url2)

    if not data:
        return []

    preds_today    = data.get("predictions", []) if data else []
    preds_tomorrow = data2.get("predictions", []) if data2 else []

    if len(preds_today) < 2 and len(preds_tomorrow) < 2:
        return []

    # Parse into absolute minutes since midnight local today
    now_utc   = datetime.now(timezone.utc)
    now_local = now_utc - timedelta(hours=4)  # EDT
    now_mins  = now_local.hour * 60 + now_local.minute
    today     = now_local.date()

    events = []
    for p in preds_today + preds_tomorrow:
        try:
            t = dt.strptime(p["t"], "%Y-%m-%d %H:%M")
            delta_days = (t.date() - today).days
            mins = delta_days * 1440 + t.hour * 60 + t.minute
            val  = float(p["v"])
            # Deduplicate — only add if not already present at this time
            if not any(abs(e[0] - mins) < 5 for e in events):
                events.append((mins, val))
        except:
            pass

    if len(events) < 2:
        return []

    events.sort(key=lambda x: x[0])

    # Find overall min/max across all events for normalization
    all_vals = [e[1] for e in events]
    mn, mx = min(all_vals), max(all_vals)
    rng = mx - mn if mx - mn > 0.1 else 1.0

    # Build 24 points, each 1 hour apart, starting from now
    curve = []
    for i in range(24):
        target_mins = now_mins + (i * 60)

        # Find the two hi/lo events surrounding this time
        before = None
        after  = None
        for j in range(len(events) - 1):
            m0, v0 = events[j]
            m1, v1 = events[j+1]
            if m0 <= target_mins <= m1:
                before = (m0, v0)
                after  = (m1, v1)
                break

        if before and after:
            span = after[0] - before[0]
            frac = (target_mins - before[0]) / span if span > 0 else 0
            frac = max(0.0, min(1.0, frac))
            # Cosine interpolation — natural tidal shape
            interp = before[1] + (after[1] - before[1]) * (1 - math.cos(frac * math.pi)) / 2
        else:
            # Outside known range — use nearest endpoint
            if target_mins < events[0][0]:
                interp = events[0][1]
            else:
                interp = events[-1][1]

        norm = round((interp - mn) / rng, 3)
        curve.append(norm)

    # Exaggerate vertical range so low tide = 0.0, high tide = 1.0 always
    # This ensures the wave shape is always visible on the LED panel
    if curve:
        mn = min(curve)
        mx = max(curve)
        rng = mx - mn if mx - mn > 0.01 else 1.0
        curve = [round((v - mn) / rng, 3) for v in curve]

    print(f"Tide {station}: now={now_mins//60:.0f}h{now_mins%60:.0f}m curve[0]={curve[0]:.2f} curve[1]={curve[1]:.2f} events={[(int(e[0]//60),round(e[1],1)) for e in events[:6]]}")
    return curve

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
        results = await asyncio.gather(
            fetch_marine(client, spot),
            fetch_wind(client, spot),
            fetch_tide_now(client, spot["tide_station"]),
            fetch_tide_hilo(client, spot["tide_station"]),
            fetch_tide_curve(client, spot["tide_station"]),
            fetch_water_temp(client, spot),
            fetch_ndbc(client, spot["ndbc_buoy"]) if spot["ndbc_buoy"] else asyncio.sleep(0),
            return_exceptions=True
        )

    marine, wind, tide_now, tide_hilo, tide_curve, water_temp, buoy = results

    # Wave height — prefer NDBC buoy for Pacific spots
    buoy_data = buoy if isinstance(buoy, dict) and buoy else None
    if buoy_data and buoy_data.get("wave_height"):
        off_h = buoy_data["wave_height"]
        off_p = buoy_data.get("wave_period") or 8.0
        off_d = int(buoy_data.get("wave_direction") or 270)
        print(f"Using NDBC buoy: {off_h}m {off_p}s {off_d}deg")
    elif isinstance(marine, dict):
        curr  = marine.get("current", {})
        off_h = curr.get("wave_height") or 0.0
        off_p = curr.get("wave_period") or 6.0
        off_d = int(curr.get("wave_direction") or 90)
    else:
        raise HTTPException(502, "Marine data unavailable")

    # Tomorrow from Open-Meteo hourly
    if isinstance(marine, dict) and "hourly" in marine:
        wh = marine["hourly"]["wave_height"]
        wp = marine["hourly"]["wave_period"]
        wd = marine["hourly"]["wave_direction"]
        idx = min(33, len(wh)-1)
        tmrw_h = wh[idx] or 0.0
        tmrw_p = wp[idx] or 6.0
        tmrw_d = int(wd[idx] or off_d)
    else:
        tmrw_h = off_h; tmrw_p = off_p; tmrw_d = off_d

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
    tide_pts = tide_curve if isinstance(tide_curve, list) else []
    wtemp = water_temp if isinstance(water_temp, (int,float)) else 0

    # Compute surf
    ht_ft   = nearshore_ft(off_h, off_p, off_d, spot)
    tmrw_ft = nearshore_ft(tmrw_h, tmrw_p, tmrw_d, spot)
    stars   = calc_stars(ht_ft, off_p, off_d, wind_dir, wind_mph, spot)
    cond    = cond_label(stars, ht_ft)
    wlabel  = wind_label(wind_dir, spot["orientation"])
    wqual   = wind_quality(wind_dir, spot["orientation"])

    disp_lo = round(ht_ft * 0.8, 1) if ht_ft >= 0.5 else 0.0
    disp_hi = round(ht_ft * 1.25, 1) if ht_ft >= 0.5 else 1.0

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
        "tmrw_ht":      tmrw_ft,
        "tmrw_ht_lo":   round(tmrw_ft * 0.8, 1) if tmrw_ft >= 0.5 else 0.0,
        "tmrw_ht_hi":   round(tmrw_ft * 1.25, 1) if tmrw_ft >= 0.5 else 1.0,
        "tmrw_flat":    tmrw_ft < 0.5,
        "tide_now":     round(tide_cur, 2),
        "tide_min":     round(tide_mn, 2),
        "tide_max":     round(tide_mx, 2),
        "tide_curve":   tide_pts,
        "water_temp":   round(float(wtemp), 1),
        "updated":      datetime.utcnow().isoformat() + "Z",
    }
