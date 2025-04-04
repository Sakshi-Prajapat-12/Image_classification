import streamlit as st 
import tensorflow
import numpy as np 
from PIL import Image

# load model

from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image
model = load_model('C:\\Users\\sakshi prajapat\\Desktop\\growtech\\CatDogImagePredictions.h5')


# class label 
class_label = ['Cat','Dog']

# UI
st.title("Dog and Cat Image Prediction")
st.write("Please provie an image to classify it is Dog of Cat ")

# file uploader
uploaded_file = st.file_uploader("Upload Your file",type=['jpg','png','jpeg'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img,caption="Uploaded Image")

    # preproces the image

    img = img.resize(size = (150,150))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr,axis = 0)
    img_arr/=255.0

    # prediction

    prediction = model.predict(img_arr)
    predicted_class = class_label[int(prediction[0][0])]

    #st.write("Prediction",predicted_class)
    st.button("Predict",predicted_class)