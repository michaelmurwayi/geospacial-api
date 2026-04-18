from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from api.serializers import SuitabilitySerializer

from services.suitability_service import SuitabilityService
from services.ml_predictor import MLPredictor
from services.temperature_service import TemperatureService




# Create your views here.

class SuitabilityPredictView(APIView):

     def post(self, request):
        serializer = SuitabilitySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        predictor = MLPredictor()
        result = predictor.predict(data)

        return Response(result)
     
class LocationDataView(APIView):

    def post(self, request):
        latitude = request.data.get("latitude")
        longitude = request.data.get("longitude")

        # ---------------- VALIDATION ----------------
        if latitude is None or longitude is None:
            return Response(
                {"error": "latitude and longitude are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            latitude = float(latitude)
            longitude = float(longitude)
        except ValueError:
            return Response(
                {"error": "latitude and longitude must be numbers"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ---------------- SERVICE CALL ----------------
        service = TemperatureService(latitude, longitude)
        temperature_data = service.get_temperature_summary()

        # ---------------- RESPONSE ----------------
        data = {
            "latitude": latitude,
            "longitude": longitude,
            "temperature": temperature_data,  # ✅ now it's JSON serializable
        }

        return Response(data, status=status.HTTP_200_OK)