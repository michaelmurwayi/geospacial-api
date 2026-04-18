import requests
from datetime import date, timedelta
from collections import defaultdict
from statistics import mean


class RainfallService:
    BASE_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    BASE_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, latitude: float, longitude: float):
        self.lat = latitude
        self.lng = longitude

    # -------------------------------
    # PUBLIC METHOD
    # -------------------------------
    def get_rainfall_summary(self):
        return {
            "latitude": self.lat,
            "longitude": self.lng,
            "current_rainfall": self._get_current_rainfall(),
            "weekly_total": self._get_weekly_rainfall(),
            "monthly_rainfall": self._get_monthly_breakdown(),
            "yearly_range": self._get_yearly_range(),
        }

    # -------------------------------
    # CURRENT RAINFALL
    # -------------------------------
    def _get_current_rainfall(self):
        params = {
            "latitude": self.lat,
            "longitude": self.lng,
            "hourly": "precipitation",
            "forecast_days": 1,
            "timezone": "auto",
        }

        res = requests.get(self.BASE_FORECAST_URL, params=params)
        data = res.json()

        hourly = data.get("hourly", {})
        precipitation = hourly.get("precipitation", [])

        # latest hour rainfall
        return precipitation[-1] if precipitation else 0

    # -------------------------------
    # WEEKLY TOTAL (last 7 days)
    # -------------------------------
    def _get_weekly_rainfall(self):
        end_date = date.today()
        start_date = end_date - timedelta(days=7)

        rainfall = self._get_daily_rainfall(start_date, end_date)
        return round(sum(rainfall), 2) if rainfall else 0

    # -------------------------------
    # MONTH NAME MAP
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
    # MONTHLY BREAKDOWN (12 MONTHS)
    # -------------------------------
    def _get_monthly_breakdown(self):
        end_date = date.today()
        start_date = end_date - timedelta(days=365)

        params = {
            "latitude": self.lat,
            "longitude": self.lng,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "precipitation_sum",
            "timezone": "auto",
        }

        res = requests.get(self.BASE_ARCHIVE_URL, params=params)

        if res.status_code != 200:
            return []

        data = res.json().get("daily", {})

        rainfall = data.get("precipitation_sum", [])
        dates = data.get("time", [])

        monthly_data = defaultdict(list)

        # Group rainfall by month
        for i in range(len(dates)):
            year, month = dates[i].split("-")[:2]
            key = f"{year}-{month}"

            monthly_data[key].append(rainfall[i])

        # Convert to structured output
        results = []
        for key in sorted(monthly_data.keys()):
            year, month_num = key.split("-")

            results.append({
                "month": self.MONTH_NAMES.get(month_num, month_num),
                "year": int(year),
                "total_rainfall": round(sum(monthly_data[key]), 2),  # total rainfall per month
                "avg_daily_rainfall": round(mean(monthly_data[key]), 2),  # optional
            })

        return results

    # -------------------------------
    # YEAR RANGE
    # -------------------------------
    def _get_yearly_range(self):
        end_date = date.today()
        start_date = end_date - timedelta(days=365)

        rainfall = self._get_daily_rainfall(start_date, end_date)

        if not rainfall:
            return {"min": 0, "max": 0}

        return {
            "min": round(min(rainfall), 2),
            "max": round(max(rainfall), 2),
        }

    # -------------------------------
    # HELPER: DAILY RAINFALL
    # -------------------------------
    def _get_daily_rainfall(self, start_date, end_date):
        params = {
            "latitude": self.lat,
            "longitude": self.lng,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "precipitation_sum",
            "timezone": "auto",
        }

        res = requests.get(self.BASE_ARCHIVE_URL, params=params)

        if res.status_code != 200:
            return []

        data = res.json().get("daily", {})

        return data.get("precipitation_sum", [])