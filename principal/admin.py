from django.contrib import admin
from . import models

@admin.register(models.Projects)
class ProjectsAdmin(admin.ModelAdmin):
    pass

# Register your models here.
