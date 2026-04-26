from django.urls import path
from .views import SuitabilityPredictView, LocationDataView

urlpatterns = [
    path("suitability/", SuitabilityPredictView.as_view()),
    path("location/", LocationDataView.as_view()),
]