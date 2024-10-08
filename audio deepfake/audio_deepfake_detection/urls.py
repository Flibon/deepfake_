# audio_deepfake_detection/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect  # Add this import

urlpatterns = [
    path('admin/', admin.site.urls),
    path('upload/', include('detector.urls')),  # Include app's URLs for audio upload

    # Redirect root URL to /upload/
    path('', lambda request: redirect('upload_audio')),  # Redirect empty path to upload_audio
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
