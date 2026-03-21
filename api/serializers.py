from rest_framework import serializers

class SuitabilitySerializer(serializers.Serializer):
    soil_ph = serializers.FloatField()
    rainfall = serializers.FloatField()
    temperature = serializers.FloatField()
    elevation = serializers.FloatField()
    latitude = serializers.FloatField(required=False)
    longitude = serializers.FloatField(required=False)