import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained DecisionTreeClassifier model from the pickle file
with open('decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Pulsar Classification App')
st.write('Welcome to the Pulsar Classification App! This app uses a trained Decision Tree model to predict whether an object is a pulsar or not based on selected features.')

# Sidebar for User Input
st.sidebar.title('User Input')
amplitude_asymmetry = st.sidebar.slider('Amplitude Distribution Asymmetry', min_value=-2.0, max_value=75.0, value=0.5)
amplitude_shape = st.sidebar.slider('Amplitude Distribution Shape', min_value=-2.0, max_value=10.0, value=0.5)

# Load the dataset with the selected features
columns = ['amplitude_distribution_asymmetry', 'amplitude_distribution_shape']
data = pd.DataFrame({
    'amplitude_distribution_asymmetry': [amplitude_asymmetry],
    'amplitude_distribution_shape': [amplitude_shape]
})

# Function to make predictions using the loaded model
def predict_pulsar(features):
    prediction = model.predict(features)
    return prediction[0]

# Make predictions using the user input
prediction = predict_pulsar(data)

# Display the Prediction
st.write('Predicted Class:')
if prediction == 0:
    st.write('Not Pulsar')
else:
    st.write('Pulsar :star:')
