import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained TensorFlow CNN model
model = load_model("C:\\Users\\sakshi prajapat\\Desktop\\growproject\\CatDogModel.h5")

# Define class labels (Change this based on your dataset)
class_labels = ["Cat", "Dog","Other"]

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = cv2.resize(frame, (150,150))  # Resize to model's input shape
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict the class 
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    label = class_labels[class_index]

    # Display the label on the frame
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Dog and Cat Classifier", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
