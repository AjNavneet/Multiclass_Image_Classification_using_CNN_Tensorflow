import tensorflow as tf
import numpy as np
from .admin import model_path
import tensorflow as tf
import cv2
import os

# Import necessary TensorFlow and OpenCV libraries

# Create a class for the classifier model
class Classifier:

    def __init__(self, train_dir):
        """
        Initialize the classifier model with various parameters.
        :param train_dir: The training directory containing labeled image folders.
        """
        self.label = ['driving_license', 'social_security', "others"]  # Define class labels
        self.img_size = 224  # Set the image size
        self.epochs = 10  # Number of training epochs
        self.train_dir = train_dir  # Training data directory
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            # Data augmentation settings
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=30,
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False
        )

    def train(self):
        """
        Create and train the image classification model.
        """
        model = self.model()  # Create the model
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)  # Initialize the optimizer
        model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])  # Compile the model

        train_data = self.get_data(self.train_dir)  # Get training data

        x_train = []  # Create a list for training data
        y_train = []  # Create a list for training labels

        for feature, label in train_data:
            x_train.append(feature)  # Append data to the list
            y_train.append(label)

        # Normalize the data
        x_train = np.array(x_train) / 255
        x_train = x_train.reshape(-1, self.img_size, self.img_size, 3)
        y_train = np.array(y_train)

        self.datagen.fit(x_train)  # Apply data augmentation
        history = model.fit(x_train, y_train, epochs=self.epochs)  # Train the model
        print(history)
        model.save(model_path)  # Save the trained model

    def get_data(self, data_dir):
        """
        Load and preprocess training data.
        :param data_dir: Training data directory
        :return: Numpy array of image and class pairs
        """
        data = []
        for each_label in self.label:
            path = os.path.join(data_dir, each_label)
            class_num = self.label.index(each_label)
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]
                    resized_arr = cv2.resize(img_arr, (self.img_size, self.img_size))
                    data.append([resized_arr, class_num])
                except Exception as e:
                    print(e)
        return np.array(data)

    def model(self):
        """
        Define the architecture of the convolutional neural network (CNN) model.
        :return: The model object
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=(224, 224, 3))
        model.add(tf.keras.layers.MaxPool2D())

        model.add(tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"))
        model.add(tf.keras.layers.MaxPool2D())

        model.add(tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"))
        model.add(tf.keras.layers.MaxPool2D())

        model.add(tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"))
        model.add(tf.keras.layers.MaxPool2D())
        model.add(tf.keras.layers.Dropout(0.4))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dense(3, activation="softmax"))  # Three output classes

        return model
