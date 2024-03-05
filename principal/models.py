from django.db import models
from django.conf import settings

# Create your models here.
class Projects(models.Model):
    title = models.CharField(max_length=25, verbose_name="title")
    section = models.CharField(max_length=15, verbose_name="section")
    description = models.TextField(verbose_name="description")
    pub_date = models.DateField(db_comment="Date and time when the article was published", verbose_name="Publication Date")
    img = models.ImageField(upload_to = settings.MEDIA_ROOT / "uploads/projects/", verbose_name="Cover Image")
  
    
    class Meta:
        verbose_name = "project"
        verbose_name_plural = "projects"
        
      
    def __str__(self):
        return self.title