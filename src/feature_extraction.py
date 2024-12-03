import os
import librosa
import numpy as np
import pandas as pd

import data_preprocessing

def extract_features(file_path):
    try:
        # File Path Check
        print(f"Processing file: {file_path}")

        # Load the audio file
        audio, sr = librosa.load(file_path, sr=None)
        
    # Extract various audio features
        # Chroma features
        chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr)
        
        # RMS Energy
        rms = librosa.feature.rms(y=audio)
        
        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        
        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        
        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        
        # Zero Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        
        # Harmony and Perceptr (these are more complex)
        harmony, perceptr = librosa.effects.hpss(audio)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio)
        
        # MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        
        # Compute mean and variance for each feature
        features = [
            # Filename (optional, you might want to add this separately)
            # Length of audio
            len(audio),
            
            # Chroma STFT
            np.mean(chroma_stft), np.var(chroma_stft),
            
            # RMS
            np.mean(rms), np.var(rms),
            
            # Spectral Centroid
            np.mean(spectral_centroid), np.var(spectral_centroid),
            
            # Spectral Bandwidth
            np.mean(spectral_bandwidth), np.var(spectral_bandwidth),
            
            # Spectral Rolloff
            np.mean(spectral_rolloff), np.var(spectral_rolloff),
            
            # Zero Crossing Rate
            np.mean(zero_crossing_rate), np.var(zero_crossing_rate),
            
            # Harmony and Perceptr
            np.mean(harmony), np.var(harmony),
            np.mean(perceptr), np.var(perceptr),
            
            # Tempo
            tempo,
        ]
        
        # Add MFCC means and variances
        for i in range(20):
            features.extend([np.mean(mfccs[i]), np.var(mfccs[i])])
        
        return features
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def save_features(extract_features):
    features = []
    labels = []
    filenames = []

    # Print out the genres and data directory
    print("Genres:", data_preprocessing.genres)
    print("Data Directory:", data_preprocessing.data_dir)

    for genre in data_preprocessing.genres:
        genre_path = os.path.join(data_preprocessing.data_dir, genre)
        print(f"Checking genre path: {genre_path}")

        # Find all audio files
        for root, dirs, files in os.walk(genre_path):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.au', '.aiff', '.flac')):
                    file_path = os.path.join(root, file)
                    feature = extract_features(file_path)
                    
                    if feature is not None:
                        features.append(feature)
                        labels.append(genre)
                        filenames.append(file)
    
    print(f"Number of features extracted: {len(features)}")
    print(f"Number of labels: {len(labels)}")                    

    # Create DataFrame with all features, including filename
    column_names = [
        'length',
        'chroma_stft_mean', 'chroma_stft_var',
        'rms_mean', 'rms_var',
        'spectral_centroid_mean', 'spectral_centroid_var',
        'spectral_bandwidth_mean', 'spectral_bandwidth_var',
        'rolloff_mean', 'rolloff_var',
        'zero_crossing_rate_mean', 'zero_crossing_rate_var',
        'harmony_mean', 'harmony_var',
        'perceptr_mean', 'perceptr_var',
        'tempo'
    ]
    
    # Add MFCC columns
    for i in range(1, 21):
        column_names.extend([f'mfcc{i}_mean', f'mfcc{i}_var'])
    
    column_names.append('label')
    
    # Create DataFrame
    df = pd.DataFrame(features, columns=column_names[:-1])
    df['label'] = labels
    
    # Ensure directory exists
    os.makedirs('data/features', exist_ok=True)
    
    # Save to CSV
    df.to_csv("data/features/features.csv", index=False)
    print("Features saved to data/features/features.csv")

    return features, labels

# Call the function
features, labels = save_features(extract_features)