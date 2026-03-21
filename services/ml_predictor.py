# app/services/ml_predictor.py
import requests
import json
import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    Predictor for coffee suitability using Ollama LLM.
    Returns a structured JSON with score, limiting factors, suitability, and explanation.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "phi3:mini",
        timeout: int = 60,
        max_retries: int = 2
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs prediction + structured explanation via Ollama.
        """

        prompt = f"""
You are an expert in coffee agronomy.

Arabica coffee grows best under:
- Soil pH: 5.5 – 6.5
- Rainfall: 1200 – 1800 mm/year
- Temperature: 15 – 24°C
- Elevation: 1200 – 2200 m

Evaluate the following conditions:
Soil pH: {data['soil_ph']}
Rainfall: {data['rainfall']} mm/year
Temperature: {data['temperature']} °C
Elevation: {data['elevation']} m

Return STRICT JSON ONLY with the following keys:
- "score": integer 0–100
- "limiting_factors": list of strings
- "suitability": "Suitable" or "Not Suitable"
- "explanation": short explanation

Do NOT include any text outside the JSON object.
""".strip()

        for attempt in range(1, self.max_retries + 2):
            try:
                logger.info("Ollama prediction attempt %d/%d", attempt, self.max_retries + 1)
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                raw_text = response.json().get("response", "").strip()

                # Extract JSON safely
                match = re.search(r'\{.*\}', raw_text, re.DOTALL)
                if match:
                    raw_text = match.group()

                result = json.loads(raw_text)

                # Clean values
                result["score"] = int(result["score"]) if result.get("score") is not None else None
                result["limiting_factors"] = result.get("limiting_factors", [])
                result["suitability"] = result.get("suitability", "Unknown")
                result["explanation"] = result.get("explanation", "")

                return result

            except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
                logger.warning("Ollama attempt %d failed: %s", attempt, str(e))

        # fallback if all attempts fail
        logger.error("Ollama failed after all retries. Returning fallback values.")
        return {
            "score": None,
            "limiting_factors": [],
            "suitability": "Unknown",
            "explanation": "Ollama API unavailable or model failed to load. Check system resources and try again."
        }

    def warmup(self) -> bool:
        """
        Warm up the model for faster future responses.
        """
        try:
            payload = {
                "model": self.model,
                "prompt": "warmup",
                "stream": False
            }
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            response.raise_for_status()
            logger.info("Ollama warmup successful")
            return True
        except requests.RequestException as e:
            logger.warning("Ollama warmup failed: %s", str(e))
            return False