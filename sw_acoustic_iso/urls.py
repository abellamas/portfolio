from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='sw_acoustic_iso'),
    path('export_excel/', views.export_excel, name='export_excel')
]
