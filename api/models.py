from django.db import models
from django.contrib.postgres.fields import JSONField  # or use models.JSONField for Django 4+


# Create your models here.

class Suitability(models.Model):
    id = models.AutoField(primary_key=True, unique=True)
    soil_ph = models.FloatField()
    rainfall = models.FloatField()
    temperature = models.FloatField()
    elevation = models.FloatField()

    prediction = models.CharField(max_length=50)

    created_at = models.DateTimeField(auto_now_add=True)

class SuitabilityLog(models.Model):
    """
    Logs every coffee suitability prediction with input, result, and explanation.
    """
    created_at = models.DateTimeField(auto_now_add=True)
    location_name = models.CharField(max_length=255, blank=True, null=True)  # optional
    latitude = models.FloatField(blank=True, null=True)
    longitude = models.FloatField(blank=True, null=True)
    input_data = models.JSONField()
    prediction = models.JSONField()
    ai_explanation = models.JSONField()

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Coffee Suitability Log"
        verbose_name_plural = "Coffee Suitability Logs"

    def __str__(self):
        loc = self.location_name or f"({self.latitude}, {self.longitude})"
        return f"Prediction at {loc} on {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}"