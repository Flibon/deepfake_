# detector/audio_processing.py

import librosa
import numpy as np

def extract_features(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # Return the mean of the MFCCs across the time axis
    return np.mean(mfccs.T, axis=0)
