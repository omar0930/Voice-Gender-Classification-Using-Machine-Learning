# Voice Gender Classification Using Machine Learning

## Overview
The **Voice Gender Classification** project aims to classify voice recordings as either male or female using machine learning techniques. The system extracts relevant audio features and applies classification models to achieve high accuracy.

## Features
- Extracts features from audio recordings (e.g., pitch, frequency, formants)
- Trains multiple machine learning models
- Evaluates model performance using accuracy metrics
- Provides a user-friendly interface for classification

## Installation
Clone the repository using:
```bash
git clone https://github.com/omar0930/Voice-Gender-Classification-Using-Machine-Learning.git
cd Voice-Gender-Classification-Using-Machine-Learning
```


## Dataset
The dataset consists of labeled voice recordings containing male and female speech samples. Features such as fundamental frequency, harmonic-to-noise ratio, and formants are extracted from the audio data.

## Workflow
1. Load and preprocess the dataset.
2. Extract relevant audio features.
3. Split the dataset into training and testing sets.
4. Train different machine learning models (e.g., SVM, Random Forest, Neural Networks).
5. Evaluate model performance using accuracy, precision, and recall.
6. Deploy the trained model for real-time voice classification.

## Results
The model was trained and tested on a labeled dataset, achieving the following performance metrics:
- **Support Vector Machine (SVM):** 91.2% accuracy
- **Random Forest Classifier:** 93.5% accuracy
- **Neural Network:** 95.8% accuracy

These results indicate that the neural network model performed the best in classifying gender from voice recordings. Further improvements can be made by using larger datasets and fine-tuning model hyperparameters.

## Technologies Used
- Python
- NumPy & Pandas
- Librosa (for audio processing)
- Scikit-learn
- TensorFlow/Keras (for deep learning models)
