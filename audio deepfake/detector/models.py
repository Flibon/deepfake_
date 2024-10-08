# detector/models.py

from django.db import models

class AudioFile(models.Model):
    file = models.FileField(upload_to='audio/')  # Save audio files to the 'media/audio/' directory

    def __str__(self):
        return self.file.name
