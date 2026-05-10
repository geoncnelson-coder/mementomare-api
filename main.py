"""
Memento Mare API
FastAPI server that fetches surf data, runs wave model, returns clean JSON to ESP32
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import math
import asyncio
from datetime import datetime

app = FastAPI(title="Memento Mare API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# SPOT DATABASE — add spots here, no reflashing needed
# ============================================================
SPOTS = {
    "washout": {
        "name": "WASHOUT",
        "lat": 32.646, "lon": -79.941,
        "orientation": 100,
        "tide_station": "8665530",
        "Ks": 1.495,       # Green's Law shoaling (NOAA FBPS1 buoy at 4m)
        "near_depth": 4.0, # nearshore depth meters
    },
    "iop": {
        "name": "IOP",
        "lat": 32.787, "lon": -79.771,
        "orientation": 95,
        "tide_station": "8665530",
        "Ks": 1.414,
        "near_depth": 5.0,
    },
    "huntington": {
        "name": "HUNTINGTON",
        "lat": 33.654, "lon": -118.003,
        "orientation": 270,
        "tide_station": "9410660",
        "Ks": 1.495,
        "near_depth": 4.0,
    },
    "blacks": {
        "name": "BLACKS",
        "lat": 32.856, "lon": -117.253,
        "orientation": 270,
        "tide_station": "9410170",
        "Ks": 1.622,       # Scripps Canyon energy focusing bonus
        "near_depth": 6.0,
    },
    "pipeline": {
        "name": "PIPELINE",
        "lat": 21.665, "lon": -158.053,
        "orientation": 330,
        "tide_station": "1612340",
        "Ks": 1.778,       # Steep reef, extreme shoaling
        "near_depth": 2.0,
    },
}

# ============================================================
# WAVE PHYSICS
# ============================================================
def refraction_kr(theta_deg: float, d_offshore: float, d_nearshore: float) -> float:
    """Snell's Law refraction coefficient"""
    if theta_deg > 89:
        theta_deg = 89
    C_off  = math.sqrt(9.81 * d_offshore)
    C_near = math.sqrt(9.81 * d_nearshore)
    sin_t  = math.sin(math.radians(theta_deg)) * (C_near / C_off)
    sin_t  = min(sin_t, 0.999)
    cos_off  = math.cos(math.radians(theta_deg))
    cos_near = math.sqrt(1 - sin_t ** 2)
    if cos_near < 0.01:
        cos_near = 0.01
    return math.sqrt(cos_off / cos_near)

def swell_angle(swell_dir: int, orientation: int) -> float:
    """Angle between swell and shore normal"""
    shore_normal = (orientation + 180) % 360
    diff = abs(swell_dir - shore_normal)
    if diff > 180:
        diff = 360 - diff
    return float(diff)

def period_factor(period: float) -> float:
    """Long period waves shoal more aggressively"""
    if period >= 14: return 1.15
    if period >= 10: return 1.08
    if period >= 8:  return 1.03
    return 1.0

def nearshore_height_ft(offshore_m: float, period: float, swell_dir: int, spot: dict) -> float:
    """Full nearshore height using Green's Law + Snell's Law"""
    Hs    = offshore_m * spot["Ks"]
    theta = swell_angle(swell_dir, spot["orientation"])
    Kr    = refraction_kr(theta, 20.0, spot["near_depth"])
    Pf    = period_factor(period)
    H_near = Hs * Kr * Pf
    return round(H_near * 3.28084, 2)  # meters to feet

def wind_label(wind_dir: int, orientation: int) -> str:
    offshore_center = (orientation + 180) % 360
    diff = abs(wind_dir - offshore_center)
    if diff > 180:
        diff = 360 - diff
    if diff <= 45:  return "OFFSHORE"
    if diff <= 75:  return "SIDE-OFF"
    if diff <= 105: return "SIDE-ON"
    return "ONSHORE"

def wind_quality(wind_dir: int, orientation: int) -> str:
    label = wind_label(wind_dir, orientation)
    if label in ("OFFSHORE", "SIDE-OFF"): return "good"
    if label == "SIDE-ON": return "marginal"
    return "bad"

def calc_stars(ht: float, period: float, swell_dir: int, wind_dir: int, wind_mph: float, spot: dict) -> int:
    if ht < 0.5: return 0
    s = 0
    if ht >= 3.0: s += 2
    elif ht >= 1.5: s += 1
    wl = wind_label(wind_dir, spot["orientation"])
    if wl in ("OFFSHORE", "SIDE-OFF") and wind_mph < 20: s += 1
    if period >= 10: s += 1
    return min(s, 4)

def cond_label(stars: int, ht: float) -> str:
    if ht < 0.5:   return "FLAT"
    if stars >= 4:  return "EPIC"
    if stars >= 3:  return "GOOD"
    if stars >= 2:  return "FAIR"
    return "POOR"

# ============================================================
# API FETCHERS
# ============================================================
async def fetch_ndbc_buoy(client: httpx.AsyncClient, buoy_id: str) -> dict:
    """Fetch latest obs from NOAA NDBC buoy — real offshore swell data"""
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{buoy_id}.txt"
    r = await client.get(url, timeout=10)
    r.raise_for_status()
    lines = r.text.strip().split("\n")
    # Line 0 = header names, Line 1 = units, Line 2+ = data (most recent first)
    if len(lines) < 3:
        return {}
    headers = lines[0].split()
    data    = lines[2].split()  # most recent observation
    obs = dict(zip(headers, data))
    def safe(key):
        v = obs.get(key, "MM")
        try: return float(v) if v != "MM" else None
        except: return None
    return {
        "wave_height":    safe("WVHT"),   # meters
        "wave_period":    safe("DPD"),    # dominant period seconds
        "wave_direction": safe("MWD"),    # degrees
    }

async def fetch_marine(client: httpx.AsyncClient, spot: dict) -> dict:
    url = (
        f"https://marine-api.open-meteo.com/v1/marine"
        f"?latitude={spot['lat']}&longitude={spot['lon']}"
        f"&current=wave_height,wave_period,wave_direction"
        f"&hourly=wave_height,wave_period,wave_direction"
        f"&forecast_days=2&timezone=America%2FNew_York"
    )
    r = await client.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

async def fetch_wind(client: httpx.AsyncClient, spot: dict) -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={spot['lat']}&longitude={spot['lon']}"
        f"&current=wind_speed_10m,wind_direction_10m"
        f"&hourly=wind_speed_10m,wind_direction_10m"
        f"&wind_speed_unit=mph&forecast_days=2"
        f"&timezone=America%2FNew_York"
    )
    r = await client.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

async def fetch_tide_current(client: httpx.AsyncClient, station: str) -> float:
    url = (
        f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        f"?date=latest&station={station}&product=water_level"
        f"&datum=MLLW&time_zone=lst_ldt&units=english&format=json"
    )
    r = await client.get(url, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data: return 3.0
    return float(data[-1]["v"])

async def fetch_tide_range(client: httpx.AsyncClient, station: str) -> tuple:
    url = (
        f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        f"?date=today&station={station}&product=predictions&interval=hilo"
        f"&datum=MLLW&time_zone=lst_ldt&units=english&format=json"
    )
    r = await client.get(url, timeout=10)
    r.raise_for_status()
    preds = r.json().get("predictions", [])
    if not preds: return 0.0, 6.0
    vals = [float(p["v"]) for p in preds]
    return min(vals), max(vals)

async def fetch_tide_curve(client: httpx.AsyncClient, station: str) -> list:
    """Fetch next 24 hours of hourly tide predictions from NOW forward, normalized 0-1"""
    from datetime import timezone, timedelta
    now = datetime.now(timezone.utc)
    end = now + timedelta(hours=24)
    begin_str = now.strftime("%Y%m%d %H:%M")
    end_str   = end.strftime("%Y%m%d %H:%M")
    url = (
        f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        f"?begin_date={begin_str}&end_date={end_str}"
        f"&station={station}&product=predictions&interval=h"
        f"&datum=MLLW&time_zone=gmt&units=english&format=json"
    )
    r = await client.get(url, timeout=10)
    r.raise_for_status()
    preds = r.json().get("predictions", [])
    if not preds: return []
    vals = [float(p["v"]) for p in preds]
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx - mn > 0.1 else 1.0
    return [round((v - mn) / rng, 3) for v in vals]

async def fetch_water_temp(client: httpx.AsyncClient, spot: dict) -> float:
    url = (
        f"https://marine-api.open-meteo.com/v1/marine"
        f"?latitude={spot['lat']}&longitude={spot['lon']}"
        f"&current=sea_surface_temperature"
        f"&temperature_unit=fahrenheit"
    )
    r = await client.get(url, timeout=10)
    r.raise_for_status()
    return r.json()["current"]["sea_surface_temperature"]

# ============================================================
# ENDPOINTS
# ============================================================
@app.get("/")
def root():
    return {
        "service": "Memento Mare API",
        "spots": list(SPOTS.keys()),
        "usage": "/surf/{spot_id}"
    }

@app.get("/spots")
def get_spots():
    return [{"id": k, "name": v["name"]} for k, v in SPOTS.items()]

@app.get("/surf/{spot_id}")
async def get_surf(spot_id: str):
    spot = SPOTS.get(spot_id.lower())
    if not spot:
        raise HTTPException(status_code=404, detail=f"Spot '{spot_id}' not found")

    async with httpx.AsyncClient(verify=False) as client:
        marine_task      = fetch_marine(client, spot)
        wind_task        = fetch_wind(client, spot)
        tide_now_task    = fetch_tide_current(client, spot["tide_station"])
        tide_range_task  = fetch_tide_range(client, spot["tide_station"])
        tide_curve_task  = fetch_tide_curve(client, spot["tide_station"])
        water_temp_task  = fetch_water_temp(client, spot)
        # NDBC buoy if configured
        buoy_task = (fetch_ndbc_buoy(client, spot["ndbc_buoy"])
                     if spot.get("ndbc_buoy") else asyncio.sleep(0))

        try:
            marine, wind, tide_now, tide_range, tide_curve, water_temp, buoy = await asyncio.gather(
                marine_task, wind_task, tide_now_task, tide_range_task,
                tide_curve_task, water_temp_task, buoy_task,
                return_exceptions=True
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    # Handle exceptions from gather
    if isinstance(marine, Exception):
        raise HTTPException(status_code=502, detail=f"Marine API error: {marine}")
    if isinstance(wind, Exception):
        raise HTTPException(status_code=502, detail=f"Wind API error: {wind}")

    # Parse marine — prefer NDBC buoy for Pacific spots
    buoy_ok = (isinstance(buoy, dict) and buoy.get("wave_height") is not None)
    if buoy_ok:
        off_h = buoy["wave_height"]
        off_p = buoy.get("wave_period") or 6.0
        off_d = int(buoy.get("wave_direction") or 90)
    else:
        curr  = marine["current"] if isinstance(marine, dict) else {}
        off_h = curr.get("wave_height") or 0.0
        off_p = curr.get("wave_period") or 6.0
        off_d = int(curr.get("wave_direction") or 90)

    # Tomorrow from Open-Meteo hourly (buoy is real-time only)
    if isinstance(marine, dict) and "hourly" in marine:
        wh = marine["hourly"]["wave_height"]
        wp = marine["hourly"]["wave_period"]
        wd = marine["hourly"]["wave_direction"]
        idx = min(33, len(wh)-1)
        tmrw_h = wh[idx] or 0.0
        tmrw_p = wp[idx] or 6.0
        tmrw_d = wd[idx] or 90
    else:
        tmrw_h = off_h; tmrw_p = off_p; tmrw_d = off_d

    # Parse wind
    wind_mph = wind["current"].get("wind_speed_10m") or 0.0
    wind_dir = wind["current"].get("wind_direction_10m") or 0
    ws = wind["hourly"]["wind_speed_10m"]
    wdh = wind["hourly"]["wind_direction_10m"]
    tmrw_wind_mph = ws[idx] or 0.0
    tmrw_wind_dir = wdh[idx] or 0

    # Tide
    tide_min, tide_max = tide_range if not isinstance(tide_range, Exception) else (0.0, 6.0)
    tide_cur = tide_now if not isinstance(tide_now, Exception) else 3.0
    tide_pts = tide_curve if not isinstance(tide_curve, Exception) else []
    wtemp = water_temp if not isinstance(water_temp, Exception) else 0.0

    # Wave model
    ht_ft    = nearshore_height_ft(off_h, off_p, off_d, spot)
    tmrw_ft  = nearshore_height_ft(tmrw_h, tmrw_p, tmrw_d, spot)
    stars    = calc_stars(ht_ft, off_p, off_d, wind_dir, wind_mph, spot)
    cond     = cond_label(stars, ht_ft)
    wlabel   = wind_label(wind_dir, spot["orientation"])
    wquality = wind_quality(wind_dir, spot["orientation"])

    return {
        "spot":        spot["name"],
        "ht":          ht_ft,
        "ht_lo":       round(ht_ft * 0.8, 1),
        "ht_hi":       round(ht_ft * 1.25, 1),
        "period":      off_p,
        "swell_dir":   off_d,
        "condition":   cond,
        "stars":       stars,
        "wind_mph":    round(wind_mph, 1),
        "wind_dir":    wind_dir,
        "wind_label":  wlabel,
        "wind_quality": wquality,
        "tmrw_ht":     tmrw_ft,
        "tmrw_ht_lo":  round(tmrw_ft * 0.8, 1),
        "tmrw_ht_hi":  round(tmrw_ft * 1.25, 1),
        "tide_now":    round(tide_cur, 2),
        "tide_min":    round(tide_min, 2),
        "tide_max":    round(tide_max, 2),
        "tide_curve":  tide_pts,
        "water_temp":  round(wtemp, 1),
        "tmrw_flat":   tmrw_ft < 0.5,
        "updated":     datetime.utcnow().isoformat() + "Z",
    }
