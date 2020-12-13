import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
import time

# Import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.optimizers import Adam
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

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
        self.fit(x=data.train.images, y=data.train.labels, epochs=4, batch_size=128)


class Main(blackboard.Blackboard):

    def __init__(self):
        super(Main, self).__init__()
        self.model = ModelSequential()
        self.model.start_model()
        self.model.train_the_model()
        result = self.model.evaluate(x=data.test.images, y=data.test.labels)
        print("{0}: {1:.2%}".format(self.model.metrics_names[1], result[1]))

    def plot_images(self, images, cls_true, cls_pred=None):
        assert len(images) == len(cls_true) == 9

        # Create figure with 3x3 sub-plots.
        fig, axes = plt.subplots(3, 3)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        for i, ax in enumerate(axes.flat):
            # Plot image.
            ax.imshow(images[i].reshape((28, 28)), cmap='binary')

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true[i])
            else:
                xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
        plt.show()

    def others_events(self, event, cursor):
        if event.type == pg.MOUSEBUTTONDOWN:
            if cursor.colliderect(self.boton1.rect):
                self.name = "image%s.png" % time.strftime("%Y-%m-%d %H:%M:%S")
                # self.capture(self.screen, self.name, (50, 50), (500, 500))



                # Intento 1
                # image = ImageSettings()
                # img_array = image.image_to_matrix(self.name)
                # img_array.flatten()
                # img_array.Rescaling(1./255)
                # print('imagen')
                # print('imagen')
                # print('imagen')
                # print(img_array)
                # print(img_array.shape)
                #
                # predictions = self.model.predict(x=[img_array])
                # print('predicciones: ')
                # print(np.argmax(predictions[0]))
                # print(predictions[0])


            elif cursor.colliderect(self.boton2.rect):
                self.screen.fill((119, 119, 119))
                self.screen.blit(self.blackboard_draw, (50, 50))
            elif cursor.colliderect(self.boton3.rect):
                images = data.test.images[15:24]
                cls_true = data.test.cls[15:24]
                y_pred = self.model.predict(x=images)
                cls_pred = np.argmax(y_pred, axis=1)

                self.plot_images(images=images, cls_true=cls_true, cls_pred=cls_pred)


if __name__ == '__main__':
    pizarra = Main()
    pizarra.show_blackboard()
