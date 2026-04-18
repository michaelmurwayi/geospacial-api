# app/services/suitability_service.py
import logging
from typing import Dict, Any
from django.utils import timezone
from .ml_predictor import MLPredictor
from .llm_explainer import OllamaExplainer
from  api.models import SuitabilityLog  # import model

logger = logging.getLogger(__name__)


class SuitabilityService:
    def __init__(self):
        self.predictor = MLPredictor(
            base_url="http://127.0.0.1:11434",
            model="phi3:mini",
            timeout=60,
            max_retries=2
        )
        self.explainer = OllamaExplainer(
            base_url="http://127.0.0.1:11434",
            model="phi3:mini",
            timeout=60,
            max_retries=2
        )

        if self.explainer.health_check():
            self.explainer.warmup()

    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Run prediction
        prediction = self.predictor.predict(input_data)

        # Default explanation
        ai_explanation = {
            "summary": "AI explanation not generated.",
            "recommendations": [],
            "risk_level": "unknown"
        }

        # Generate AI explanation
        if self.explainer.health_check():
            try:
                ai_explanation = self.explainer.explain_suitability(
                    features=input_data,
                    prediction=prediction
                )
            except Exception as e:
                logger.warning("Ollama explanation failed: %s", str(e))

        # Save to database
        try:
            SuitabilityLog.objects.create(
                created_at=timezone.now(),
                location_name=input_data.get("location_name"),
                latitude=input_data.get("latitude"),
                longitude=input_data.get("longitude"),
                input_data=input_data,
                prediction=prediction,
                ai_explanation=ai_explanation
            )
        except Exception as e:
            logger.error("Failed to save suitability log: %s", str(e))

        return {
            "prediction": prediction,
            "ai_explanation": ai_explanation
        }