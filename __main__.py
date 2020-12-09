import matplotlib.pyplot as plt
import numpy as np
import math

# Import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam
from tensorflow.examples.tutorials.mnist import input_data


# Import other classes
from images_settings import ImageSettings
import blackboard

data = input_data.read_data_sets('data/MNIST/', one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)


class ModelSequential(Sequential):
    """Defines the model object inheriting from models.Sequential"""

    def __init__(self):
        """Constructor del objeto ModelSequential"""
        Sequential.__init__(self)

        # New attributes
        self.image_setting = ImageSettings(28, 10)

    def start_model(self):
        """Define the necessary structure"""
        self.add(InputLayer(input_shape=(self.image_setting.img_size_flat,)))  # Add an input layer
        self.add(Reshape(self.image_setting.img_shape_full))  # convolutional layers expect images with shape (28, 28, 1), so we reshape
        self.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu',
                        name='layer_conv1'))  # First convolutional layer
        self.add(MaxPooling2D(pool_size=2, strides=2))
        self.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu',
                        name='layer_conv2'))  # Second convolutional layer
        self.add(MaxPooling2D(pool_size=2, strides=2))
        self.add(
            Flatten())  # Flatten the 4-level output of convolutional layers to 2-rank that can be entered into a fully connected layer
        self.add(Dense(128, activation='relu'))  # Fully connected first layer
        self.add(Dense(self.image_setting.num_classes,
                       activation='softmax'))  # last fully connected layer from which the classification is derived

        # compile the model
        optimizer = Adam(lr=1e-3)
        self.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def train_the_model(self):
        self.fit(x=data.train.images, y=data.train.labels, epochs=10, batch_size=128)


if __name__ == '__main__':
    print('jajaja')
    print('jajaja')
    print('jajaja')
    print(data.train.images)
    model = ModelSequential()
    model.start_model()
    model.train_the_model()
    result = model.evaluate(x=data.test.images, y=data.test.labels)
    print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))

