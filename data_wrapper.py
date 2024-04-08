"""
A data wrapper module to process the Handwritten Hangul dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class DataWrapper:
    def __init__(self, raw_data_path: str, output_path: str, rand_seed: int):
        """
        DataWrapper class initialisation
        """

        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.rand_seed = rand_seed

        self.train_size = 0.8
        self.val_size = 0.5 # 0.5 * 0.2 = 0.1

        self.image_names = []
        self.labels = []
        self.classes = None
        
        self.df = None
        self.test_df = None
        self.train_df = None

        self.x_full = None
        self.y_full = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

        self.__extract_dataframe()
        self.__write_csv()
        self.__load_csv()

        self.__load_data()
    
    def __extract_dataframe(self):
        """
        Private helper function to extract raw data from file path to dataframe
        """
    
        # Loop through files in directory and file path and character
        for image_name in os.listdir(self.raw_data_path):
            if (image_name.endswith(".jpg")):
                self.image_names.append(image_name)
                self.labels.append(image_name.split("_")[0])
        
        self.classes = pd.unique(self.labels)
        
        # Do a train/test split and create dataframes
        X_train, X_test, y_train, y_test = train_test_split(self.image_names, self.labels, train_size = self.train_size, random_state = self.rand_seed, stratify = self.labels)
        self.df = pd.DataFrame({"image_name" : self.image_names, "labels" : self.labels})
        self.train_df = pd.DataFrame({"image_name" : X_train, "labels" : y_train})
        self.test_df = pd.DataFrame({"image_name" : X_test, "labels" : y_test})
    
    def __write_csv(self):
        """
        Private helper function to write extracted data to csv
        """
        
        self.df.to_csv(str(self.output_path) + "\\handwritten_hangul.csv", index=False)
        self.train_df.to_csv(str(self.output_path) +  "\\train_handwritten_hangul.csv", index=False)
        self.test_df.to_csv(str(self.output_path) +  "\\test_handwritten_hangul.csv", index=False)

    def __load_csv(self):
        """
        Private helper function to load already existing csv
        """
        self.df = pd.read_csv(str(self.output_path) + "\\handwritten_hangul.csv")
        self.train_df = pd.read_csv(str(self.output_path) + "\\train_handwritten_hangul.csv")
        self.test_df = pd.read_csv(str(self.output_path) + "\\test_handwritten_hangul.csv")

    def __load_data(self):
        """
        Private helper function to load data into arrays and perform a train/test/val split
        """

        self.x_full = self.load_images(self.image_names)
        self.y_full = np.array(self.labels)
        self.x_train, self.x_test, y_train, y_test = train_test_split(self.x_full,
                                                                        self.y_full,
                                                                        train_size=self.train_size,
                                                                        random_state=self.rand_seed,
                                                                        stratify=self.y_full)
        
        self.x_test, self.x_val, y_test, y_val = train_test_split(self.x_test,
                                                                    y_test,
                                                                    train_size=self.val_size,
                                                                    random_state=self.rand_seed,
                                                                    stratify=self.y_test)
        
        self.y_train = self.__encode(y_train)
        self.y_val = self.__encode(y_val)
        self.y_test = self.__encode(y_test)
    
    def __encode(self, array: np.ndarray) -> np.ndarray:
        """
        Private helper function to encode a one-hot vector of the labels
        """

        le = LabelEncoder()
        le.fit(self.classes)
        
        return le.transform(array)

    
    def load_image(self, image_name: str) -> np.ndarray:
        """
        Function to load an image from file name and return as array
        """

        image = np.array(mpimg.imread(self.raw_data_path + "\\" + image_name))

        return(image / 255.0)
    
    def load_images(self, image_names: list) -> np.ndarray:
        """
        Function to load multiple images fromn file name and return as an array
        """

        images = []
        for image_name in image_names:
            images.append(self.load_image(image_name))

        return np.array(images)
        
    
    def plot_image(self, image_name: str, scale = False, axis = False):
        """
        Function to load and plot an image
        """

        image = self.load_image(image_name=image_name)

        if scale:
            plt.rcParams["figure.figsize"] = (image.shape[0]/28.0, image.shape[1]/28.0)
        if not axis:
            plt.axis("off")
        plt.imshow(image, cmap="gray", interpolation="nearest")
        plt.show()