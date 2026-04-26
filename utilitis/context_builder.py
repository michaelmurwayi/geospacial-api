class SuitabilityContextBuilder:

    def build(self, data):

        temp = data["temperature"]
        rain = data["rainfall"]
        geo = data["geo"]
        veg = data["vegetation"]

        return {
            "latitude": data["latitude"],
            "longitude": data["longitude"],

            # temperature
            "current_temp": temp["current_temperature"],
            "temp_avg": temp["weekly_average"],
            "temp_min": temp["yearly_range"]["min"],
            "temp_max": temp["yearly_range"]["max"],

            # rainfall
            "rain_current": rain["current_rainfall"],
            "rain_weekly": rain["weekly_total"],
            "rain_min": rain["yearly_range"]["min"],
            "rain_max": rain["yearly_range"]["max"],

            # geo
            "elevation": geo["elevation"],
            "soil": geo["soil_type"],

            # vegetation
            "ndvi": veg.get("ndvi"),
            "land_cover": veg.get("land_cover"),
        }