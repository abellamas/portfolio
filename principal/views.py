from django.shortcuts import render
from . import models
# Create your views here.
def index(request):
    projects_obj = models.Projects.objects.all()
    ctx = {
        "projects" : projects_obj,
    }
    return render(request, "index.html", ctx)
