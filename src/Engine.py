# Import necessary classes and functions from the ML_Pipeline package
from ML_Pipeline.training import Classifier
from ML_Pipeline.inference import Inference
from ML_Pipeline.admin import train_dir

### Training ###

# Create a Classifier object for training the model
train_object = Classifier(train_dir)  # Instantiate the Classifier class, passing the 'train_dir' parameter
train_object.train()  # Train the machine learning model

### Inference ###

# Define the file path of the test image
filename = "../input/Data/Testing_Data/driving_license/1.jpg" 

# Create an Inference object for making predictions
infer_object = Inference()  # Instantiate the Inference class
response = infer_object.infer(filename)  # Perform inference on the specified image

# Print the result of the inference
print("The result is : ", response)  # Display the output of the inference
