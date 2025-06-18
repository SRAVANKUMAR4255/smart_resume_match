from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from core import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('upload/', views.upload_resumes, name='upload_resumes'),
    path('selected/', views.selected_resumes, name='selected_resumes'),
    path('download/<int:resume_id>/', views.download_resume, name='download_resume'),
    path('mark-best/<int:resume_id>/', views.mark_as_best, name='mark_as_best'),
    path('unmark-best/<int:resume_id>/', views.unmark_as_best, name='unmark_as_best'),
    path('mark-best-multiple/', views.mark_best_multiple, name='mark_best_multiple'),
    path('remove-resumes/', views.remove_resumes, name='remove_resumes'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) 