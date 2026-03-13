from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from services.suitability_service import SuitabilityService

# Create your views here.

class SuitabilityPredictView(APIView):

    def post(self, request):
        data = request.data

        result = SuitabilityService().predict(data)

        return Response(result)