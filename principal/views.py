from django.shortcuts import render
from . import models
from django.utils import translation
# Create your views here.
def index(request):
    # Internationalization 
    projects_obj = models.Projects.objects.all()
    ctx = {
        "projects" : projects_obj,
    }
    return render(request, "index.html", ctx)
