import os
import time
import math
import requests
import pandas as pd

INPUT_CSV = "ml/nyeri_points.csv"
OUTPUT_CSV = "ml/coffee_training_dataset.csv"

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"

REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS = 1.0  # be polite to APIs


def fetch_soil_ph(lat: float, lon: float):
    """
    Fetch soil pH (pH in H2O) from SoilGrids.
    Uses topsoil layer approximation from returned depths.
    """
    params = {
        "lon": lon,
        "lat": lat,
        "property": "phh2o",
        "depth": "0-5cm",
        "value": "mean"
    }

    try:
        response = requests.get(SOILGRIDS_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        layers = data.get("properties", {}).get("layers", [])
        for layer in layers:
            if layer.get("name") == "phh2o":
                depths = layer.get("depths", [])
                for depth in depths:
                    if depth.get("label") == "0-5cm":
                        ph_val = depth.get("values", {}).get("mean")
                        if ph_val is not None:
                            # SoilGrids pH often scaled by 10
                            return round(ph_val / 10.0, 2)

        return None

    except Exception as e:
        print(f"[WARN] SoilGrids failed for ({lat}, {lon}): {e}")
        return None


def fetch_climate(lat: float, lon: float):
    """
    Fetch climatology from NASA POWER.
    Returns:
    - annual rainfall (mm/year) from PRECTOTCORR monthly climatology sum
    - annual mean temperature (°C) from T2M monthly climatology mean
    """
    params = {
        "parameters": "T2M,PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "format": "JSON"
    }

    try:
        response = requests.get(NASA_POWER_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        parameter_data = data.get("properties", {}).get("parameter", {})

        t2m = parameter_data.get("T2M", {})
        precip = parameter_data.get("PRECTOTCORR", {})

        # NASA POWER climatology often returns JAN..DEC monthly averages
        month_keys = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                      "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

        temps = []
        rains = []

        for m in month_keys:
            t = t2m.get(m)
            p = precip.get(m)

            if t is not None and t != -999:
                temps.append(float(t))

            if p is not None and p != -999:
                # PRECTOTCORR is often mm/day climatology for that month
                # Approximate monthly total by multiplying by days in month
                days = days_in_month(m)
                rains.append(float(p) * days)

        annual_temp = round(sum(temps) / len(temps), 2) if temps else None
        annual_rainfall = round(sum(rains), 2) if rains else None

        return annual_temp, annual_rainfall

    except Exception as e:
        print(f"[WARN] NASA POWER failed for ({lat}, {lon}): {e}")
        return None, None


def days_in_month(month_abbr: str) -> int:
    month_days = {
        "JAN": 31, "FEB": 28, "MAR": 31, "APR": 30,
        "MAY": 31, "JUN": 30, "JUL": 31, "AUG": 31,
        "SEP": 30, "OCT": 31, "NOV": 30, "DEC": 31
    }
    return month_days[month_abbr]


def rule_based_label(soil_ph, rainfall, temperature, elevation):
    """
    Simple initial suitability label for building a starter training set.
    This is NOT final scientific labeling, but useful for MVP / bootstrapping.
    """
    score = 0

    if soil_ph is not None and 5.5 <= soil_ph <= 6.5:
        score += 25

    if rainfall is not None and 1200 <= rainfall <= 1800:
        score += 25

    if temperature is not None and 15 <= temperature <= 24:
        score += 25

    if elevation is not None and 1200 <= elevation <= 2200:
        score += 25

    return 1 if score >= 50 else 0


def build_dataset():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    required = ["lat", "lon"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    if "elevation" not in df.columns:
        df["elevation"] = None

    results = []

    for idx, row in df.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        elevation = row["elevation"]

        try:
            elevation = float(elevation) if pd.notna(elevation) else None
        except Exception:
            elevation = None

        print(f"[INFO] Processing point {idx}/{len(df)}: ({lat}, {lon})")

        soil_ph = fetch_soil_ph(lat, lon)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        temperature, rainfall = fetch_climate(lat, lon)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        suitable = rule_based_label(
            soil_ph=soil_ph,
            rainfall=rainfall,
            temperature=temperature,
            elevation=elevation
        )

        results.append({
            "lat": lat,
            "lon": lon,
            "elevation": elevation,
            "soil_ph": soil_ph,
            "rainfall": rainfall,
            "temperature": temperature,
            "suitable": suitable
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n[OK] Dataset saved to: {OUTPUT_CSV}")
    print(out_df.head())


if __name__ == "__main__":
    build_dataset()