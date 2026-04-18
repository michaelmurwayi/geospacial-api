import requests
from datetime import date, timedelta
from statistics import mean
from collections import defaultdict


class TemperatureService:
    BASE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    BASE_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, latitude: float, longitude: float):
        self.lat = latitude
        self.lng = longitude

    # -------------------------------
    # PUBLIC METHOD
    # -------------------------------
    def get_temperature_summary(self):
        return {
            "latitude": self.lat,
            "longitude": self.lng,
            "current_temperature": self._get_current_weather(),
            "weekly_average": self._get_weekly_average(),
            "monthly_temperatures": self._get_monthly_breakdown(),
            "yearly_range": self._get_yearly_range(),
        }

    # -------------------------------
    # CURRENT TEMP
    # -------------------------------
    def _get_current_weather(self):
        params = {
            "latitude": self.lat,
            "longitude": self.lng,
            "current_weather": True,
            "timezone": "auto",
        }

        res = requests.get(self.BASE_FORECAST_URL, params=params)
        data = res.json()

        return data.get("current_weather", {}).get("temperature")

    # -------------------------------
    # WEEKLY AVG (last 7 days)
    # -------------------------------
    def _get_weekly_average(self):
        end_date = date.today()
        start_date = end_date - timedelta(days=7)

        temps = self._get_daily_temperatures(start_date, end_date)
        return round(mean(temps), 2) if temps else None

    # -------------------------------
    # MONTH NAME MAPPING
    # -------------------------------
    MONTH_NAMES = {
        "01": "Jan",
        "02": "Feb",
        "03": "March",
        "04": "April",
        "05": "May",
        "06": "June",
        "07": "July",
        "08": "Aug",
        "09": "Sept",
        "10": "Oct",
        "11": "Nov",
        "12": "Dec",
    }

    # -------------------------------
    # MONTHLY BREAKDOWN (LAST 12 MONTHS)
    # -------------------------------
    def _get_monthly_breakdown(self):
        end_date = date.today()
        start_date = end_date - timedelta(days=365)

        params = {
            "latitude": self.lat,
            "longitude": self.lng,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": "auto",
        }

        res = requests.get(self.BASE_ARCHIVE_URL, params=params)

        if res.status_code != 200:
            return []

        data = res.json().get("daily", {})

        max_temps = data.get("temperature_2m_max", [])
        min_temps = data.get("temperature_2m_min", [])
        dates = data.get("time", [])

        monthly_data = defaultdict(list)

        # Group by month
        for i in range(len(dates)):
            year, month = dates[i].split("-")[:2]
            key = f"{year}-{month}"

            avg_temp = (max_temps[i] + min_temps[i]) / 2
            monthly_data[key].append(avg_temp)

        # Sort chronologically and convert names
        monthly_results = []
        for key in sorted(monthly_data.keys()):
            year, month_num = key.split("-")

            month_name = self.MONTH_NAMES.get(month_num, month_num)

            monthly_results.append({
                "month": month_name,
                "year": int(year),
                "mean_temperature": round(mean(monthly_data[key]), 2),
            })

        return monthly_results

    # -------------------------------
    # YEAR RANGE (last 12 months)
    # -------------------------------
    def _get_yearly_range(self):
        end_date = date.today()
        start_date = end_date - timedelta(days=365)

        temps = self._get_daily_temperatures(start_date, end_date)

        if not temps:
            return {"min": None, "max": None}

        return {
            "min": round(min(temps), 2),
            "max": round(max(temps), 2),
        }

    # -------------------------------
    # HELPER: FETCH DAILY TEMPS
    # -------------------------------
    def _get_daily_temperatures(self, start_date, end_date):
        params = {
            "latitude": self.lat,
            "longitude": self.lng,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": "auto",
        }

        res = requests.get(self.BASE_ARCHIVE_URL, params=params)

        if res.status_code != 200:
            return []

        data = res.json().get("daily", {})

        max_temps = data.get("temperature_2m_max", [])
        min_temps = data.get("temperature_2m_min", [])

        temps = [
            (max_temps[i] + min_temps[i]) / 2
            for i in range(min(len(max_temps), len(min_temps)))
        ]

        return temps