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

filename2="label_encoder.pk"
with open(path.join(filename2),'rb') as t:
    label_encoder=joblib.load(t)
    
# Function to extract features from an audio file
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    y, sr = librosa.load(file_path, mono=True)
    features = []
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        features.extend(mfccs)
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        features.extend(chroma)
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
        features.extend(mel)
    return features

# Function to predict whether an audio file contains a scream or not
def predict_audio(file_path, model, label_encoder):
    feature = extract_features(file_path)
    feature = np.array(feature).reshape(1, -1)  # Ensure it's 2D

    prediction = model.predict(feature)

    if len(prediction) == 0:
        return "Unknown"

    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

#upload audio file
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])

st.audio(audio_file, format=audio_file.type)  # Play uploaded audio


# Add a "Predict" button
if st.button("Predict"):
    prediction_result = predict_audio(audio_file, model, label_encoder)
    
    if prediction_result == 1:
        st.write("Nearby Officer is alerted")
    else:
        st.write("No alert Required")
