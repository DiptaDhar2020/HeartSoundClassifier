import streamlit as st
from keras.models import load_model
import librosa
import numpy as np

# Disclaimer Text
disclaimer = """
**Disclaimer: This heart sound classification app is a tool designed to assist in the preliminary assessment of heart sounds. It is not a substitute for a professional medical diagnosis. The app uses an artificial neural network (ANN) model with approximately 95% accuracy, but its accuracy may vary depending on the quality of the heart sound recording and other factors. If you have concerns about your heart health, please consult a healthcare professional for a proper evaluation and diagnosis.
"""
resources = """
Additional Resources:
  - Bangladesh Cardiac Society: https://banglacardio.org/
  - National Heart Foundation of Bangladesh: https://www.nhf.org.bd/
"""

# Copyright Text
copyright = "©diptadhar2024"

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
# Set wide mode for the app
st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; margin-top:-15px;'>Heart Sound Classification App</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a heart sound wav file", type="wav")

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav", start_time=0)

    # Classify heartbeat
    prediction = classify_heartbeat(uploaded_file)

    if prediction > 80:
        st.success("Normal Heartbeat.")
    else:
        st.warning("Abnormal Heartbeat.")

# Display Disclaimer and Copyright
st.markdown(f"<center style='margin-top: 30px; text-align: justify; text-align-last: center; font-size: 18px;'>{disclaimer}</center>", unsafe_allow_html=True)
st.markdown(resources, unsafe_allow_html=True)
st.markdown(f"<center style='margin-top: 20px;'>{copyright}</center>", unsafe_allow_html=True)
