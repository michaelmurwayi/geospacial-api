from api.models import Suitability

class SuitabilityRepository:

    def create(self, data, prediction):

        return Suitability.objects.create(
            soil_ph=data["soil_ph"],
            rainfall=data["rainfall"],
            temperature=data["temperature"],
            elevation=data["elevation"],
            prediction=prediction
        )