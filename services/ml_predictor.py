# app/services/ml_predictor.py
import requests
import json
import re

class MLPredictor:
    """
    Predictor for coffee suitability using Ollama LLM.
    Returns a structured JSON with score, limiting factors, suitability, and explanation.
    """

    def predict(self, data):
        # 1. Build a strict JSON prompt for LLM
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
            """

        # 2. Call Ollama API
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=10
            )
            raw_text = response.json()["response"].strip()
        except Exception as e:
            # Fail gracefully
            return {
                "score": None,
                "limiting_factors": [],
                "suitability": "Unknown",
                "explanation": f"Ollama API error: {str(e)}"
            }

        # 3. Extract JSON from LLM response safely
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            raw_text = match.group()

        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError:
            # fallback: return raw text in explanation
            result = {
                "score": None,
                "limiting_factors": [],
                "suitability": "Unknown",
                "explanation": raw_text
            }

        # 4. Clean values and enforce types
        result["score"] = int(result["score"]) if result.get("score") is not None else None
        result["limiting_factors"] = result.get("limiting_factors", [])
        result["suitability"] = result.get("suitability", "Unknown")
        result["explanation"] = result.get("explanation", "")

        return result