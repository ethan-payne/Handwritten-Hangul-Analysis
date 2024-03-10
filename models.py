"""
A module to build various neural network architectures
"""

from keras import models, layers

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
