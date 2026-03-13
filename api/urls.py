from django.urls import path
from .views import SuitabilityPredictView

urlpatterns = [
    path("predict/", SuitabilityPredictView.as_view()),
]