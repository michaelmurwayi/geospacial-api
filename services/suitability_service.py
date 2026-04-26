import requests
import json
import re

from utilitis.context_builder import SuitabilityContextBuilder

class OllamaSuitabilityEngine:

    def __init__(
        self,
        base_url="http://127.0.0.1:11434",
        model="phi3:mini",
    ):
        self.base_url = base_url
        self.model = model

    def predict(self, data):

        context = SuitabilityContextBuilder().build(data)
        prompt = self._build_prompt(context)

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )

        raw = response.json().get("response", "")

        return self._extract_json(raw)

    def _build_prompt(self, c):

        return f"""
            You are an agricultural suitability expert for Arabica coffee in Kenya.

            Use ONLY the structured data below.

            DATA:

            Location:
            - Lat: {c['latitude']}
            - Lon: {c['longitude']}

            Temperature:
            - Current: {c['current_temp']} °C
            - Avg: {c['temp_avg']} °C
            - Range: {c['temp_min']} - {c['temp_max']}

            Rainfall:
            - Current: {c['rain_current']} mm
            - Weekly: {c['rain_weekly']} mm
            - Range: {c['rain_min']} - {c['rain_max']}

            Geography:
            - Elevation: {c['elevation']} m
            - Soil: {c['soil']}

            Vegetation:
            - NDVI: {c['ndvi']}
            - Land cover: {c['land_cover']}

            TASK:
            Evaluate suitability for coffee farming.

            Return STRICT JSON ONLY:

            {{
            "score": 0-100,
            "suitability": "Highly Suitable | Moderately Suitable | Not Suitable",
            "limiting_factors": ["..."],
            "reason": "short explanation"
            }}
            """.strip()

    def _extract_json(self, text):
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            return {
                "score": None,
                "suitability": "Unknown",
                "limiting_factors": [],
                "reason": "Invalid LLM response"
            }

        return json.loads(match.group())