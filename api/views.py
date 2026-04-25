from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from api.serializers import SuitabilitySerializer

from services.suitability_service import SuitabilityService
from services.ml_predictor import MLPredictor
from services.temperature_service import TemperatureService
from services.rainfall_service import RainfallService
from services.geo_service import GeoSpatialService
from services.vegetation_service import VegetationService



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
      temperature_service = TemperatureService(latitude, longitude)
      temperature_data = temperature_service.get_temperature_summary()

      rainfall_service = RainfallService(latitude, longitude)
      rainfall_data = rainfall_service.get_rainfall_summary()

      geo_service = GeoSpatialService(latitude, longitude)
      geo_data = geo_service.get_geospatial_summary()

      vegetation_service = VegetationService(latitude, longitude)
      vegetation_data = vegetation_service.get_vegetation_summary()

      

      # ---------------- RESPONSE ----------------
      data = {
            "latitude": latitude,
            "longitude": longitude,
            "temperature": temperature_data,  # ✅ now it's JSON serializable
            "rainfall": rainfall_data,
            "geo": geo_data,
            "vegetation": vegetation_data
      }

      return Response(data, status=status.HTTP_200_OK)