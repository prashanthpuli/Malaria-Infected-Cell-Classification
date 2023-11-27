# Malaria Infected Cell Classification


## Overview
This project aims to classify cell images as either infected with malaria or uninfected using a deep learning model. The model is built using TensorFlow and Keras, and a Streamlit application is provided for easy image classification.


## Project Structure


- `train_model.py`: Python script for training the deep learning model.
- `app.py`: Streamlit application for image classification.
- `malaria_model.h5`: Pre-trained model file.
- `requirements.txt`: List of dependencies.
- `readme.txt`: Instructions and additional information.


## Model Architecture
The model architecture is a Convolutional Neural Network (CNN) with the following layers:
- Conv2D (32 filters, 3x3 kernel, ReLU activation)
- MaxPooling2D (2x2)
- Conv2D (64 filters, 3x3 kernel, ReLU activation)
- MaxPooling2D (2x2)
- Flatten
- Dense (128 units, ReLU activation)
- Dense (1 unit, Sigmoid activation)


## Directory Structure


- `Dataset` (place your dataset here)
  - `cell_images`
    - `Train`
      - `uninfected`
      - `parasite`
- `malaria_model.h5`
- `train_model.py`
- `app.py`
- `requirements.txt`
- `readme.txt`


## Instructions to Run


1. Install dependencies using `pip install -r requirements.txt`.
2. Run `python train_model.py` to train the model.
3. After training, run `streamlit run app.py` to start the Streamlit application.
4. Upload cell images through the application to get predictions.


## Additional Notes
- Make sure to have a properly structured dataset in the `Dataset/cell_images` directory.
- Adjust model parameters and hyperparameters in `train_model.py` as needed.