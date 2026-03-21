# app/services/suitability_service.py
import logging
from typing import Dict, Any
from .ml_predictor import MLPredictor
from .llm_explainer import OllamaExplainer

logger = logging.getLogger(__name__)


class SuitabilityService:
    """
    Handles coffee suitability predictions and AI explanations.
    """

    def __init__(self):
        # ML predictor
        self.predictor = MLPredictor(
            base_url="http://127.0.0.1:11434",
            model="phi3:mini",   # small model for Kali
            timeout=60,
            max_retries=2
        )

        # Ollama explanation helper
        self.explainer = OllamaExplainer(
            base_url="http://127.0.0.1:11434",
            model="phi3:mini",
            timeout=60,
            max_retries=2
        )

        # Warmup model once at startup (optional)
        if self.explainer.health_check():
            self.explainer.warmup()
        else:
            logger.warning("Ollama not available at startup; AI explanations will be skipped.")

    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns combined ML prediction and AI explanation.
        """
        # 1️⃣ Run ML prediction
        prediction = self.predictor.predict(input_data)

        # 2️⃣ Initialize default AI explanation
        ai_explanation: Dict[str, Any] = {
            "summary": "AI explanation not generated.",
            "recommendations": [],
            "risk_level": "unknown"
        }

        # 3️⃣ Call Ollama for explanation if available
        if self.explainer.health_check():
            try:
                ai_explanation = self.explainer.explain_suitability(
                    features=input_data,
                    prediction=prediction
                )
            except Exception as e:
                logger.warning("Ollama explanation failed: %s", str(e))

        else:
            logger.warning("Ollama unavailable; skipping AI explanation.")

        # 4️⃣ Return combined result
        return {
            "prediction": prediction,
            "ai_explanation": ai_explanation
        }