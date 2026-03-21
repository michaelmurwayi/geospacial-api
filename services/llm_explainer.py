import json
import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class OllamaExplainer:
    """
    Production-friendly wrapper for Ollama local API.
    Features:
    - health check
    - warmup
    - timeout handling
    - retries
    - fallback response
    - optional JSON parsing
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "phi3:mini",
        timeout: int = 60,
        max_retries: int = 2,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def health_check(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error("Ollama health check failed: %s", str(e))
            return False

    def warmup(self) -> bool:
        """Warm the model so first real request is faster."""
        try:
            payload = {
                "model": self.model,
                "prompt": "warmup",
                "stream": False
            }
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info("Ollama warmup successful for model=%s", self.model)
            return True
        except requests.RequestException as e:
            logger.warning("Ollama warmup failed: %s", str(e))
            return False

    def generate_text(self, prompt: str) -> str:
        """
        Generate plain text response with retries and fallback.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        for attempt in range(1, self.max_retries + 2):
            try:
                logger.info("Calling Ollama (attempt %d/%d)", attempt, self.max_retries + 1)
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()

                data = response.json()
                result = data.get("response", "").strip()

                if not result:
                    raise ValueError("Empty response from Ollama")

                return result

            except (requests.RequestException, ValueError) as e:
                logger.warning("Ollama request failed on attempt %d: %s", attempt, str(e))

        logger.error("Ollama failed after all retries. Returning fallback text.")
        return (
            "The suitability analysis was completed successfully, but the AI explanation "
            "service is currently unavailable. Please review the numeric suitability score "
            "and agronomic indicators."
        )

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """
        Ask model to return JSON only. Attempts to parse response safely.
        Returns fallback JSON if parsing fails.
        """
        strict_prompt = f"""
You must respond ONLY with valid JSON. Do not include markdown, code fences, or extra text.

{prompt}
""".strip()

        text = self.generate_text(strict_prompt)

        try:
            # Clean common markdown wrappers if model ignores instruction
            cleaned = text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned.replace("```json", "", 1).strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.replace("```", "", 1).strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

            return json.loads(cleaned)

        except json.JSONDecodeError as e:
            logger.error("Failed to parse Ollama JSON response: %s | raw=%s", str(e), text)
            return {
                "summary": "AI explanation unavailable in structured format.",
                "recommendations": [
                    "Review the suitability score and environmental variables manually."
                ],
                "risk_level": "unknown"
            }

    def explain_suitability(self, features: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate structured explanation for coffee suitability results.
        """
        prompt = f"""
You are an agricultural AI assistant helping explain coffee suitability results.

Input environmental data:
{json.dumps(features, indent=2)}

Prediction results:
{json.dumps(prediction, indent=2)}

Return ONLY valid JSON in this exact structure:
{{
  "summary": "2-4 sentence explanation of suitability",
  "recommendations": [
    "recommendation 1",
    "recommendation 2",
    "recommendation 3"
  ],
  "risk_level": "low | moderate | high"
}}
""".strip()

        return self.generate_json(prompt)