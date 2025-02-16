import pickle
import streamlit as st
from os import path
import numpy as np
import librosa
import tempfile
import joblib

st.title("Human Scream Detection App")

filename="model.pk"
with open(path.join(filename),'rb') as f:
    model=pickle.load(f)

if model is None:
    raise ValueError("Model is not loaded properly!")

if not hasattr(model, "predict"):
    raise ValueError("Loaded object is not a valid ML model!")

filename2="label_encoder.pk"
with open(path.join(filename2),'rb') as t:
    label_encoder=joblib.load(t)
    
# Function to extract features from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)  # Load audio file
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Extract MFCC features
    return np.mean(mfccs, axis=1)  # Take mean of MFCCs

# Function to predict whether an audio file contains a scream or not
def predict_audio(file_path, model, label_encoder):

    feature = extract_features(file_path)
    feature = np.array(feature).reshape(1, -1)
    prediction = model.predict(feature)
    
    if len(prediction) == 0:
        return "Unknown"

    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

#upload audio file
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])


# Add a "Predict" button
if st.button("Predict"):
    prediction_result = predict_audio(audio_file, model, label_encoder)
    
    if prediction_result == 1:
          print("Nearby Officer is alerted")
    else:
        print("No alert required")
