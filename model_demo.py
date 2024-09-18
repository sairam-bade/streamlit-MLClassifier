import streamlit as st
import pickle
import numpy as np
import requests
from io import BytesIO

@st.cache_data()
@st.cache_resource()

# Function to download the model from the internet
def load_model_from_url():
    url = "https://github.com/Ruban2205/Iris_Classification/blob/main/model.pkl?raw=true"
    response = requests.get(url)
    model = pickle.load(BytesIO(response.content))
    return model

# Load the pre-trained model
model = load_model_from_url()

# # Load the model from a local file
# def load_model():
#     with open("iris_classifier.pkl", "rb") as f:
#         model = pickle.load(f)
#     return model

# # Load the pre-trained model
# model = load_model()

# Slider input
# # Streamlit app title
# st.title("Iris Flower Classifier")

# # Get user input for the features (sepal length, sepal width, petal length, petal width)
# st.header("Input Features")

# sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
# sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
# petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.5)
# petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# # Make a prediction
# input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
# prediction = model.predict(input_data)

# # Display the prediction
# st.write(f"Predicted class: {prediction[0]}")

# Text box input
# # Streamlit app title
st.title("Iris Flower Classifier")

# Get user input for the features (sepal length, sepal width, petal length, petal width)
st.header("Input Features")

# Text input for features
sepal_length = st.text_input("Sepal Length (cm)", value="5.0")
sepal_width = st.text_input("Sepal Width (cm)", value="3.0")
petal_length = st.text_input("Petal Length (cm)", value="1.5")
petal_width = st.text_input("Petal Width (cm)", value="0.2")

# Convert inputs to float
try:
    sepal_length = float(sepal_length)
    sepal_width = float(sepal_width)
    petal_length = float(petal_length)
    petal_width = float(petal_width)
except ValueError:
    st.error("Please enter valid numbers for all inputs.")
    st.stop()

# Submit button to make prediction
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Display the prediction
    st.write(f"Predicted class: {prediction[0]}")

    # Optionally, you can also display the input values
    st.write(f"Sepal Length: {sepal_length}")
    st.write(f"Sepal Width: {sepal_width}")
    st.write(f"Petal Length: {petal_length}")
    st.write(f"Petal Width: {petal_width}")