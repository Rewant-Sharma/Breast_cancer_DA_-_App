# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import joblib
from sklearn.datasets import load_breast_cancer

# Load the pre-trained model from a pickle file
model = joblib.load('best_model_Ann.pkl')

# Load the breast cancer dataset to get feature names
breast_cancer = load_breast_cancer()
all_feature_names = breast_cancer.feature_names

# Define the indices of the top 10 features selected by SelectKBest
# Note: These indices should be replaced with the actual ones from your model training process
selected_feature_indices = [0, 2, 3, 7, 8, 20, 21, 23, 27, 28]  # Example indices
selected_feature_names = [all_feature_names[i] for i in selected_feature_indices]

# Set up the Streamlit app
st.title('Breast Cancer Prediction App')

# Add a description of the app
st.write("""
This app predicts whether a breast mass is benign or malignant based on the measurements it receives as input.
Only the most important features, as determined by our feature selection process, are included.
""")

# Create input fields for each selected feature
input_features = []
for feature in selected_feature_names:
    value = st.number_input(f"Enter {feature}", value=0.0, format="%.6f")
    input_features.append(value)

# Make prediction when user clicks the button
if st.button('Predict'):
    # Reshape input features for model prediction
    input_array = np.array(input_features).reshape(1, -1)
    
    # Make prediction and get probabilities
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)

    # Display prediction result
    st.subheader('Prediction:')
    if prediction[0] == 0:
        st.write('The breast mass is predicted to be: Benign')
    else:
        st.write('The breast mass is predicted to be: Malignant')

    # Display prediction probabilities
    st.subheader('Prediction Probability:')
    st.write(f'Benign: {probability[0][0]:.2f}')
    st.write(f'Malignant: {probability[0][1]:.2f}')

# Add feature importance information
st.subheader('Feature Importance')
st.write('This model uses the following features, listed in order of importance:')
for i, feature in enumerate(selected_feature_names, 1):
    st.write(f"{i}. {feature}")

# Add a disclaimer in the sidebar
st.sidebar.header('Disclaimer')
st.sidebar.write("""
This app is for educational purposes only and should not be used for actual medical diagnosis. 
Always consult with a qualified healthcare professional for medical advice and diagnoses.
""")