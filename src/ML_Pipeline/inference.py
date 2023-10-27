import tensorflow as tf
import numpy as np
from .admin import model_path
from tensorflow.keras.preprocessing import image

# Create an inference class
class Inference:

    def __init__(self):
        self.img_size = 224  # Set the image size for model input
        self.model = self.load_model()  # Load the pre-trained model
        self.label = ['driving_license', 'social_security', "others"]  # Define class labels

    def load_model(self):
        """Loads a pre-trained model from the specified model path"""
        model = tf.keras.models.load_model(model_path)  # Load the model using TensorFlow
        return model

    def infer(self, filename):
        """
        Loads an image, resizes it to 224x224, expands the dimensions to create a 4D tensor,
        and then makes predictions using the loaded model.
        """
        img1 = image.load_img(filename, target_size=(self.img_size, self.img_size))  # Load and resize the image
        Y = image.img_to_array(img1)  # Convert the image to a NumPy array
        X = np.expand_dims(Y, axis=0)  # Add an extra dimension to create a 4D tensor
        val = np.argmax(self.model.predict(X))  # Make predictions using the model
        class_predicted = self.label[int(val)]  # Map the predicted class index to the label
        return class_predicted  # Return the predicted class label
