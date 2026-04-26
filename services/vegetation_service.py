import ee

ee.Initialize(project="linux-devops-class")


class VegetationService:

    def __init__(self, latitude: float, longitude: float):
        self.lat = latitude
        self.lng = longitude

    # -------------------------------
    # PUBLIC METHOD
    # -------------------------------
    def get_vegetation_summary(self):
        return {
            "latitude": self.lat,
            "longitude": self.lng,
            "land_cover": self._get_land_cover(),
            "ndvi": self._get_ndvi(),
        }

    # -------------------------------
    # LAND COVER (ESA WorldCover)
    # -------------------------------
    def _get_land_cover(self):
        try:
            point = ee.Geometry.Point([self.lng, self.lat])

            dataset = ee.Image("ESA/WorldCover/v100")
            value = dataset.sample(point, 10).first().get("Map").getInfo()

            LAND_COVER_MAP = {
                10: "Tree cover",
                20: "Shrubland",
                30: "Grassland",
                40: "Cropland",
                50: "Built-up",
                60: "Bare / sparse vegetation",
                70: "Snow / ice",
                80: "Water bodies",
                90: "Wetland",
                95: "Mangroves",
                100: "Moss / lichen"
            }

            return LAND_COVER_MAP.get(value, "Unknown")

        except Exception as e:
            print("Land cover error:", e)
            return "Unknown"

    # -------------------------------
    # NDVI (Vegetation Health)
    # -------------------------------
    def _get_ndvi(self):
        try:
            point = ee.Geometry.Point([self.lng, self.lat])

            # Sentinel-2 image collection
            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR")
                .filterBounds(point)
                .filterDate("2023-01-01", "2023-12-31")
                .sort("CLOUDY_PIXEL_PERCENTAGE")
                .first()
            )

            # NDVI = (NIR - RED) / (NIR + RED)
            ndvi = collection.normalizedDifference(["B8", "B4"]).rename("NDVI")

            value = ndvi.sample(point, 10).first().get("NDVI").getInfo()

            return round(value, 3) if value else None

        except Exception as e:
            print("NDVI error:", e)
            return None