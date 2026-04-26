import requests
import json
from utilitis.context_builder import SuitabilityContextBuilder
from utilitis.json_utils import safe_extract_json


class OllamaSuitabilityEngine:

    def __init__(self, base_url="http://127.0.0.1:11434", model="phi3:mini"):
        self.base_url = base_url
        self.model = model

    def predict(self, data):

        context = SuitabilityContextBuilder().build(data)

        prompt = f"""
        You are an expert in coffee agronomy.

        Evaluate:

        {json.dumps(context, indent=2)}

        Return ONLY valid JSON.
        Use double quotes for all keys.

        {{
        "score": number,
        "suitability": "Highly Suitable | Moderately Suitable | Not Suitable",
        "limiting_factors": ["..."],
        "reason": "short explanation"
        }}
        """

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )

        raw = response.json().get("response", "")
        return safe_extract_json(raw)