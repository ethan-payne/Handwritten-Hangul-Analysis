'''
Wrapper module to work with the Handwritten Hangul dataset.
'''

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_sample_image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class DataWrapper:
    def __init__(self, raw_data_path: str, output_path: str, train_size: float, rand_seed: int):
        """
        DataWrapper class initialisation
        """

        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.train_size = train_size
        self.rand_seed = rand_seed
        self.df = None
        self.test_df = None
        self.train_df = None

        self.__extract_data()
        self.__write_csv()
        self.__load_csv()
    
    def __extract_data(self):
        """
        Private helper function to write extract raw data from file path
        """

        image_names = []
        labels = []
    
        # Loop through files in directory and file path and character
        for image_name in os.listdir(self.raw_data_path):
            if (image_name.endswith(".jpg")):
                image_names.append(image_name)
                labels.append(image_name.split("_")[0])
        self.df = pd.DataFrame({"image_name" : image_names, "labels" : labels})

        self.__train_test_split(X = image_names, Y = labels)
    
    def __write_csv(self):
        """
        Private helper function to write extracted data to csv
        """
        
        self.df.to_csv(str(self.output_path) + "\\handwritten_hangul.csv", index = False)
        self.train_df.to_csv(str(self.output_path) +  "\\train_handwritten_hangul.csv", index = False)
        self.test_df.to_csv(str(self.output_path) +  "\\test_handwritten_hangul.csv", index = False)

    def __load_csv(self):
        """
        Private helper function to load already existing csvr
        """
        self.df = pd.read_csv(str(self.output_path) + "\\handwritten_hangul.csv")
        self.train_df = pd.read_csv(str(self.output_path) + "\\train_handwritten_hangul.csv")
        self.test_df = pd.read_csv(str(self.output_path) + "\\test_handwritten_hangul.csv")


    def __train_test_split(self, X, Y):
        """
        Private helper function to peform a train/test split
        """

        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = self.train_size, random_state = self.rand_seed, stratify = Y)
        self.train_df = pd.DataFrame({"image_name" : X_train, "labels" : y_train})
        self.test_df = pd.DataFrame({"image_name" : X_test, "labels" : y_test})
        

    
    def load_image(self, image_name: str) -> np.ndarray:
        """
        Function to load an image from file name and return as array
        """

        image = mpimg.imread(self.raw_data_path + "\\" + image_name)

        return(image)
    
    def plot_image(self, image_name: str, scale = False, axis = False):
        """
        Function to load and plot an image
        """

        # Load image
        image = self.load_image(image_name=image_name)

        # Plot image
        if scale:
            plt.rcParams["figure.figsize"] = (image.shape[0]/28.0, image.shape[1]/28.0)
        if not axis:
            plt.axis("off")
        plt.imshow(image, cmap="gray", interpolation="nearest")
        plt.show()