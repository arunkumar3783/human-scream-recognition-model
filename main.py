import pickle
import streamlit as st
from os import path
import numpy as np
import librosa
import tempfile

st.title("Human Screen Detection App")

filename="model.pk"
with open(path.join("model",filename),'rb') as f:
    model=pickle.load(f)

filename2="label_encoder.pkl"
with open(path.join("model",filename2),'rb') as l:
    label_encoder=pickle.load(l)
    
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

# Play the uploaded audio file
st.audio(audio_file, format=audio_file.type)

if audio_file is not None:
    st.audio(audio_file, format=audio_file.type)  # Play uploaded audio

    # Convert file to a temporary WAV format for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio_file.read())
        temp_file_path = temp_file.name

# Add a "Predict" button
if st.button("🔍 Predict"):
    prediction_result = predict_audio(temp_file_path, model, label_encoder)
    
    if prediction_result == 1:
          print("Nearby Officer is alerted")
    else:
        print("No alert required")