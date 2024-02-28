from django.urls import path
from django.urls import re_path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("sw_acoustic_iso", include('sw_acoustic_iso.urls'))
]