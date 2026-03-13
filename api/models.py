from django.db import models

# Create your models here.

class Suitability(models.Model):

    soil_ph = models.FloatField()
    rainfall = models.FloatField()
    temperature = models.FloatField()
    elevation = models.FloatField()

    prediction = models.CharField(max_length=50)

    created_at = models.DateTimeField(auto_now_add=True)