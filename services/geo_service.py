import requests



class GeoSpatialService:
    ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
    SOIL_URL = "https://rest.isric.org/soilgrids/v2.0/classification/query"
    VEGETATION_URL = "https://api.openlandmap.org/query/landcover"

    def __init__(self, latitude: float, longitude: float):
        self.lat = latitude
        self.lng = longitude

    # -------------------------------
    # PUBLIC METHOD
    # -------------------------------
    def get_geospatial_summary(self):
        return {
            "latitude": self.lat,
            "longitude": self.lng,
            "elevation": self._get_elevation(),
            "soil_type": self._get_soil_type(),
        }

    # -------------------------------
    # ELEVATION
    # -------------------------------
    def _get_elevation(self):
        try:
            params = {
                "locations": f"{self.lat},{self.lng}"
            }

            res = requests.get(self.ELEVATION_URL, params=params)
            data = res.json()

            return data.get("results", [{}])[0].get("elevation")

        except Exception:
            return None

    # -------------------------------
    # SOIL TYPE
    # -------------------------------
    def _get_soil_type(self):
        try:
            params = {
                "lon": self.lng,
                "lat": self.lat
            }

            res = requests.get(self.SOIL_URL, params=params)
            data = res.json()

            return data.get("wrb_class_name", "Unknown")

        except Exception:
            return "Unknown"

    # -------------------------------
    # VEGETATION COVER
    # -------------------------------
    def _get_vegetation_cover(self):
        try:
            params = {
                "lat": self.lat,
                "lon": self.lng
            }

            res = requests.get(self.VEGETATION_URL, params=params)
            data = res.json()

            return data.get("landcover", "Unknown")

        except Exception:
            return "Unknown"