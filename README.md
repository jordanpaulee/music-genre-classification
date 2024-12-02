# Automated Music Genre Classification

## **Overview**
This project aims to classify music tracks into their respective genres using machine learning and deep learning approaches. The dataset used includes audio tracks with labels for various genres (e.g., blues, classical, pop). The project extracts audio features like MFCCs and spectrograms, applies data augmentation techniques, and evaluates both traditional and deep learning models to achieve accurate genre classification.

---

## **Features**
- **Audio Preprocessing**: Handles loading, cleaning, and feature extraction from raw audio files.
- **Feature Extraction**: Utilizes Mel-Frequency Cepstral Coefficients (MFCCs), chroma features, and spectral contrast.
- **Data Augmentation**: Includes time stretching, pitch shifting, and noise addition to improve model generalization.
- **Model Comparison**: Baseline classifiers (e.g., SVM, Random Forest) vs. deep learning models (e.g., CNNs on spectrograms).
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, and confusion matrices.

---

## **Folder Structure**
music_genre_classification/  
├── data/  
│     ├── raw/ # Original dataset of audio files  
│     ├── augmented/ # Augmented audio files  
│     ├── spectrograms/ # Generated spectrogram images  
│     ├── features.csv # Extracted features for tabular models  
├── src/  
│     ├── data_preprocessing.py # Handles feature extraction  
│     ├── augmentation.py # Implements data augmentation  
│     ├── model_training.py # Trains models  
│     ├── evaluation.py # Evaluates model performance  
├── models/ # Trained models  
├── notebooks/ # Jupyter notebooks for experiments  
├── results/ # Output metrics, plots, and tables  
├── report/ # IEEE report drafts and final version  
├── requirements.txt # Required Python libraries  
└── README.md # Project overview 

---

## **Dataset**
- **Name**: GTZAN Dataset or equivalent
- **Description**: A collection of 10 music genres with 100 audio tracks per genre.
- **Source**: [Marsyas GTZAN Dataset](http://marsyas.info/downloads/datasets.html)

---

## **Installation**
### **Prerequisites**
- Python 3.9 or 3.10
- A virtual environment (optional but recommended)

### **Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/music_genre_classification.git
   cd music_genre_classification
2. Install required libraries:
    ```bash
    pip install -r requirements.txt
3. Ensure the GTZAN dataset is placed in the data/raw/ folder

### **Usage**
1. Preprocess the Data
Run the script to extract features and augment the dataset:
    ```bash
    python src/data_preprocessing.py
    python src/augmentation.py
2. Train Models:
    Train baseline and deep learning models:
   ```bash
   python src/model_training.py
3. Evaluate Models
    Evaluate and generate performance metrics:
    ```bash
    python src/evaluation.py

### Technologies Used
- Python Libraries:
    - numpy, pandas, matplotlib, seaborn
    - librosa (audio processing)
    - scikit-learn (machine learning)
    - tensorflow / pytorch (deep learning)
    - audiomentations (data augmentation)

## License  
This project is licensed under the MIT License.

## Acknowledgments
GTZAN Dataset creators
Open-source libraries and contributors