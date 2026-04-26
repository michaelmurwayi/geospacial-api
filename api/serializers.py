from rest_framework import serializers


class TemperatureSerializer(serializers.Serializer):
    current_temperature = serializers.FloatField()
    weekly_average = serializers.FloatField()
    monthly_temperatures = serializers.ListField(required=False)
    yearly_range = serializers.DictField()


class RainfallSerializer(serializers.Serializer):
    current_rainfall = serializers.FloatField()
    weekly_total = serializers.FloatField()
    monthly_rainfall = serializers.ListField(required=False)
    yearly_range = serializers.DictField()


class GeoSerializer(serializers.Serializer):
    latitude = serializers.FloatField()
    longitude = serializers.FloatField()
    elevation = serializers.FloatField()
    soil_type = serializers.CharField()


class VegetationSerializer(serializers.Serializer):
    land_cover = serializers.CharField()
    ndvi = serializers.FloatField(required=False)


class SuitabilitySerializer(serializers.Serializer):
    latitude = serializers.FloatField()
    longitude = serializers.FloatField()

    temperature = TemperatureSerializer()
    rainfall = RainfallSerializer()
    geo = GeoSerializer()
    vegetation = VegetationSerializer()