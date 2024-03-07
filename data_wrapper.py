'''
Wrapper module to work with the Handwritten Hangul dataset.
'''

import os
import pandas as pd

class DataWrapper:
    def __init__(self, raw_data_path: str, output_path: str, seed: int):
        """
        DataWrapper class initialisation
        """

        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.seed = seed
        self.df = None
        self.test_df = None
        self.train_df = None

        # Check if data has been previiusly been extracted (i.e. if csv exists)
        self.__write_csv()
        self.__load_csv()
    
    def __write_csv(self):
        """
        Private helper function to write extracted data to csv
        """

        image_names = []
        labels = []
    
        # Loop through files in directory and file path and character
        for image_name in os.listdir(self.raw_data_path):
            if (image_name.endswith(".jpg")):
                image_names.append(image_name)
                labels.append(image_name.split("_")[0])

        # Convert arrays into dataframe
        self.df = pd.DataFrame({"image_name" : image_names, "labels" : labels})
        self.df.to_csv(str(self.output_path) + "\handwritten_hangul.csv", index = False)
        

    def __load_csv(self):
        """
        Private helper function to load already existing csvr
        """
        self.df = pd.read_csv(str(self.output_path) + "\handwritten_hangul.csv")


    def __train_test_split(self):
        """
        Helper function to 
        """

    
    def load_image(self, image_name: str):
        """
        Helper function to load an image from file name and return as array
        """