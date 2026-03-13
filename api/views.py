from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from services.suitability_service import SuitabilityService
from api.serializers import SuitabilitySerializer
from services.ml_predictor import MLPredictor



# Create your views here.

class SuitabilityPredictView(APIView):

     def post(self, request):
        serializer = SuitabilitySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        predictor = MLPredictor()
        result = predictor.predict(data)

        return Response(result)