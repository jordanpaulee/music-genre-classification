import os
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from soundfile import write
import librosa
import soundfile as sf

import data_preprocessing

augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
])

def augment_audio(file_path):
    # File Check
    print(f"Processing file: {file_path}")

    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Perform augmentation
    augmented_audio = augmenter(audio, sample_rate=sr)

    return augmented_audio, sr

# Save augmented audio for inspection

#augmented, sr = augment_audio("data/raw/blues/sample1.wav")
#write("data/augmented/sample1_aug.wav", augmented, sr)


def save_augments(num_versions = 3):

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
                    try:

                        # Extract the original file's directory structure
                        genre_dir = os.path.basename(os.path.dirname(file_path))  # e.g., 'blues'
                
                        # Create the output subdirectory in the augmented folder (if it doesn't exist)
                        output_path = os.path.join("data/augmented", genre_dir)
                        os.makedirs(output_path, exist_ok=True)

                        for i in range(num_versions):
                            # Augment audio
                            augmented, sr = augment_audio(file_path)

                            # Create a custom filename for the augmented version
                            augmented_file_name = f"{os.path.splitext(file)[0]}_aug_{i+1}.wav"
                            augmented_file_path = os.path.join(output_path, augmented_file_name)
                    
                            # Save augmented audio
                            sf.write(augmented_file_path, augmented, sr)
                            print(f"Saved: {augmented_file_path}")

                            # Iterate naming iterator
                            #i = i+1    
                        
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                
                

if __name__ == "__main__":
    save_augments()