import streamlit as st
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the trained model from the pickle file
model = pickle.load(open('wine_quality_model.pkl', 'rb'))

st.title("🍷 Wine Quality Prediction")

st.write("""
### Enter the wine's characteristics to predict its quality
""")

# Create input fields for the user to provide wine characteristics
fixed_acidity = st.number_input('Fixed Acidity', min_value=0.0, max_value=20.0, value=7.4)
volatile_acidity = st.number_input('Volatile Acidity', min_value=0.0, max_value=2.0, value=0.7)
citric_acid = st.number_input('Citric Acid', min_value=0.0, max_value=2.0, value=0.0)
residual_sugar = st.number_input('Residual Sugar', min_value=0.0, max_value=20.0, value=1.9)
chlorides = st.number_input('Chlorides', min_value=0.0, max_value=1.0, value=0.076)
free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', min_value=0.0, max_value=100.0, value=11.0)
total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', min_value=0.0, max_value=400.0, value=34.0)
density = st.number_input('Density', min_value=0.99000, max_value=1.00500, value=0.9978)
pH = st.number_input('pH', min_value=2.0, max_value=4.0, value=3.51)
sulphates = st.number_input('Sulphates', min_value=0.0, max_value=2.0, value=0.56)
alcohol = st.number_input('Alcohol', min_value=8.0, max_value=15.0, value=9.4)

# Convert input data into a numpy array for model prediction
input_data = np.array([[
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
    chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
    pH, sulphates, alcohol
]])

# When the user clicks the button, make a prediction
if st.button('Predict Wine Quality'):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success('🥂 The model predicts: Good Quality Wine!')
    else:
        st.error('❌ The model predicts: Bad Quality Wine.')

