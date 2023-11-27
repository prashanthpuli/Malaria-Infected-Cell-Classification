import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('malaria_model.h5')

# Function to predict and overlay hotspots
def predict_and_overlay(uploaded_file):
    try:
        # Check if the uploaded file is an image
        image = Image.open(uploaded_file)
        img = np.array(image)

        # Resize the image
        img = cv2.resize(img, (128, 128))
        st.image(img, caption='Uploaded and Resized Image', use_column_width=True)

        img_array = np.expand_dims(img, axis=0) / 255.0

        # Get predictions for both classes
        predictions = model.predict(img_array)
        prediction_infected = predictions[0][0]
        prediction_uninfected = 1 - prediction_infected  # Since it's binary classification

        # Display predictions
        st.write(f'Probability of Malaria Infected Cell: {prediction_infected * 100:.2f}%')
        st.write(f'Probability of Uninfected Cell: {prediction_uninfected * 100:.2f}%')

    except Exception as e:
        st.error(f"Error: {e}")

# Streamlit app
st.title('Malaria Infected Cell Classification')
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write('')
    st.write('Classifying...')

    # Call the function to predict and overlay hotspots
    predict_and_overlay(uploaded_file)
