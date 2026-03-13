import requests

class MLPredictor:

    def predict(self, data):

        prompt = f"""
        You are an agricultural expert.

        Arabica coffee grows under the following conditions:

        - Soil pH: 5.5 – 6.5
        - Rainfall: 1200 – 1800 mm per year
        - Temperature: 15 – 24 °C
        - Elevation: 1200 – 2200 meters

        If ANY condition is outside the acceptable range, return "Not Suitable".
        If ALL conditions are within range, return "Suitable".

        Evaluate the following conditions:

        Soil pH: {data["soil_ph"]}
        Rainfall: {data["rainfall"]} mm
        Temperature: {data["temperature"]} C
        Elevation: {data["elevation"]} m

        Respond with ONLY:
        Suitable
        or
        Not Suitable
        """

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"].strip()