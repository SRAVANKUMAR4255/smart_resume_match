from django.db import models
import os

def resume_file_path(instance, filename):
    return f'resumes/{filename}'

class Resume(models.Model):
    file = models.FileField(upload_to=resume_file_path)
    name = models.CharField(max_length=255)
    email = models.EmailField(blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    skills = models.TextField(blank=True, null=True)
    education = models.TextField(blank=True, null=True)
    experience = models.TextField(blank=True, null=True)
    score = models.FloatField(null=True, blank=True)
    skill_score = models.FloatField(null=True, blank=True)
    education_score = models.FloatField(null=True, blank=True)
    experience_score = models.FloatField(null=True, blank=True)
    context_score = models.FloatField(null=True, blank=True)
    is_best_match = models.BooleanField(default=False)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class JobDescription(models.Model):
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Job Description {self.id}" 