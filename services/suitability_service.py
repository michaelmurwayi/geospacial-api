from services.ml_predictor import MLPredictor
from repositories.suitability_repository import SuitabilityRepository

class SuitabilityService:

    def __init__(self):
        self.model = MLPredictor()
        self.repo = SuitabilityRepository()

    def predict(self, data):

        prediction = self.model.predict(data)

        record = self.repo.create(data, prediction)

        return {
            "prediction": prediction,
            "id": record.id
        }