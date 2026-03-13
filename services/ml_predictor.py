import joblib
import os

MODEL_PATH = os.path.join("ml", "coffee_model.pkl")

class MLPredictor:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict(self, data):

        features = [[
            data["soil_ph"],
            data["rainfall"],
            data["temperature"],
            data["elevation"]
        ]]

        result = self.model.predict(features)

        return result[0]