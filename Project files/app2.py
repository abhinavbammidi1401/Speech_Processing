import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Streamlit app
st.title("Dysarthria Detection - Dynamic Model Training")

# Step 1: File Upload
st.header("Upload Audio Files")
uploaded_files = st.file_uploader("Upload Audio files (.wav)", type=["wav"], accept_multiple_files=True)

# Initialize lists for features and labels
features = []
labels = []

if uploaded_files:
    st.sidebar.header("Audio Preprocessing Options")
    show_mfcc = st.sidebar.checkbox("Show MFCCs")
    display_spectrogram = st.sidebar.checkbox("Display Spectrograms")
    
    # Iterate through uploaded files
    for file in uploaded_files:
        st.subheader(f"Processing {file.name}")
        
        # Load audio
        y, sr = librosa.load(file, sr=None)
        label = 1 if "dys" in file.name.lower() else 0  # Assume label from filename (e.g., 'dys' means dysarthria)
        labels.append(label)
        
        # MFCC Feature Extraction
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        features.append(mfcc_mean)
        
        # Visualization
        if show_mfcc:
            st.write(f"MFCC for {file.name}")
            fig, ax = plt.subplots()
            img = librosa.display.specshow(mfcc, x_axis="time", sr=sr)
            fig.colorbar(img, ax=ax)
            st.pyplot(fig)
        
        if display_spectrogram:
            st.write(f"Spectrogram for {file.name}")
            spectrogram = np.abs(librosa.stft(y))
            fig, ax = plt.subplots()
            img = librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sr, y_axis='log', x_axis='time')
            fig.colorbar(img, ax=ax)
            st.pyplot(fig)

    # Convert features and labels to NumPy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Step 2: Model Training
    st.header("Model Training")
    test_size = st.slider("Test Set Size (%)", min_value=10, max_value=50, value=20)
    
    if st.button("Train Model"):
        st.write("Training Model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
        
        # Train a Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
        
        # Display Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Dysarthric", "Dysarthric"],
                    yticklabels=["Non-Dysarthric", "Dysarthric"])
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        st.pyplot(fig)
    
    # Step 3: Download Trained Model
    st.header("Save and Download Model")
    if st.button("Save Model"):
        import joblib
        joblib.dump(clf, "trained_model.pkl")
        with open("trained_model.pkl", "rb") as f:
            st.download_button("Download Trained Model", f, file_name="trained_model.pkl")

# Footer
st.text("Developed for Dysarthria Detection")
