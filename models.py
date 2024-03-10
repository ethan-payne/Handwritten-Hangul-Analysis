"""
A module to build various neural network architectures
"""

from keras import models, layers
from datetime import datetime

class MiniVGG:
    def __init__(self, input_shape: tuple, num_labels: int):
        """
        Initialises a MiniVGG network
        """

        self.model = models.Sequential()
        self.input_shape = input_shape
        self.num_labels = num_labels

    def build(self):
        """
        Builds the MiniVGG network
        """

        self.model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=self.input_shape))
        self.model.add(layers.BatchNormalization(momentum=0.9))
        self.model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization(momentum=0.9))
        self.model.add(layers.MaxPool2D(pool_size=(2,2)))  # Reduce the image size down to 0.5
        self.model.add(layers.Dropout(rate=0.25))  # Deactivate 25% of the neurons
        self.model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization(momentum=0.9))
        self.model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization(momentum=0.9))
        self.model.add(layers.MaxPool2D(pool_size=(2,2)))  # Reduce the image size down to 0.5
        self.model.add(layers.Dropout(rate=0.25))  # Deactivate 25% of the neurons
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.BatchNormalization(momentum=0.9))
        self.model.add(layers.Dropout(rate=0.5)) 
        self.model.add(layers.Dense(self.num_labels, activation='softmax')) # 30 label classes

    def save_model(self):
        """
        Save the MiniVGG network to disk
        """
        
        dt_now = datetime.now()
        dt_str = dt_now.strftime("%d.%m.%Y_%H%M%S")

        self.model.save("Models\\MiniVGG\\" + dt_str + "_MiniVGG.keras")

    def load_model(self, file_name: str):
        """
        Loads a MiniVGG network from disk
        """

        self.model = models.load_model(file_name)