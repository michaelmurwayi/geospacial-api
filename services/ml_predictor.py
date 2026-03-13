import requests

class MLPredictor:

    def predict(self, data):
        # Step 1: Build a detailed prompt for scoring
        prompt = f"""
        You are an expert in coffee agronomy.

        Arabica coffee grows best under these conditions:
        - Soil pH: 5.5 – 6.5
        - Rainfall: 1200 – 1800 mm/year
        - Temperature: 15 – 24°C
        - Elevation: 1200 – 2200 m

        Score each factor from 0 to 25 points based on how suitable it is.
        - 25 points if the factor is ideal
        - 0 points if the factor is unsuitable
        Also list any limiting factors that fall outside the ideal range.

        Here are the conditions to evaluate:

        Soil pH: {data['soil_ph']}
        Rainfall: {data['rainfall']} mm/year
        Temperature: {data['temperature']} °C
        Elevation: {data['elevation']} m

        Return a JSON object like this:

        {{
        "score": total_score_out_of_100,
        "limiting_factors": [list of factors],
        "suitability": "Suitable" or "Not Suitable",
        "explanation": "Short explanation of why these conditions are suitable or not."
        }}
        """

        # Step 2: Send the prompt to Ollama
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )

        # Step 3: Parse the response
        raw_text = response.json()["response"].strip()

        # LLM should return JSON; attempt to parse
        import json
        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError:
            # Fallback if LLM output is not perfect
            result = {
                "score": None,
                "limiting_factors": [],
                "suitability": "Unknown",
                "explanation": raw_text
            }

        return result