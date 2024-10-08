import os
import numpy as np
import librosa
import tensorflow as tf
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# Load the trained model
model = tf.keras.models.load_model('models/deepfake_detection_model.h5')

# Define parameters
SAMPLE_RATE = 16000
N_MELS = 128
max_time_steps = 109

def extract_features(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Ensure all spectrograms have the same width
    if mel_spectrogram.shape[1] < max_time_steps:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, max_time_steps - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :max_time_steps]

    return mel_spectrogram.reshape((N_MELS, max_time_steps, 1))  # Reshape for model input

def upload_audio(request):
    if request.method == 'POST' and request.FILES['audio_file']:
        audio_file = request.FILES['audio_file']
        fs = FileSystemStorage()
        filename = fs.save(audio_file.name, audio_file)
        file_path = os.path.join('media', filename)

        # Extract features and make prediction
        features = extract_features(file_path)
        prediction = model.predict(np.array([features]))  # Add batch dimension

        # Get prediction result
        result = np.argmax(prediction, axis=1)[0]  # 0 for spoof, 1 for bonafide
        result_text = 'Spoof' if result == 0 else 'Bonafide'

        return render(request, 'result.html', {'result': result_text})

    return render(request, 'upload.html')

def index(request):
    return render(request, 'upload.html')  # Ensure this returns your upload form
