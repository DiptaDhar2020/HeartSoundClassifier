import streamlit as st
from keras.models import load_model
import librosa
import numpy as np

# Load the model
model = load_model("bin_classification.hdf5")

# Function to classify heartbeat
def classify_heartbeat(filename):
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
    predicted_label = model.predict(mfccs_scaled_features)
    return predicted_label[0][0] * 100

# Streamlit app
st.title("Heart Sound Classification App")

uploaded_file = st.file_uploader("Choose a heart sound wav file", type="wav")

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav", start_time=0)

    # Classify heartbeat
    prediction = classify_heartbeat(uploaded_file)

    if prediction > 80:
        st.success("Normal Heartbeat.")
    else:
        st.warning("Abnormal Heartbeat.")
