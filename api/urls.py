from django.urls import path
from .views import SuitabilityPredictView, LocationDataView

urlpatterns = [
    path("predict/", SuitabilityPredictView.as_view()),
    path("location/", LocationDataView.as_view()),
]