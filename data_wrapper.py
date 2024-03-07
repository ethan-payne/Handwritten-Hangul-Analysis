'''
Wrapper module to work with the Handwritten Hangul dataset.
'''

import os
import pandas as pd

class DataWrapper:
    def __init__(self, path: str, seed: int):
        """
        DataWrapper class initialisation
        """

        self.path = path
        self.seed = seed
        self.df = None
        self.test_df = None
        self.train_df = None

        # Check if data has been previiusly been extracted (i.e. if csv exists)
    
    def _write_csv(self):
        """
        Private helper function to write extracted data to csv
        """

        image_names = []
        labels = []
    
        # Loop through files in directory and file path and character
        for image_name in os.listdir(self.path):
            if (image_name.endswith(".jpg")):
                image_names.append(image_name)
                labels.append(image_name.split("_")[0])

        # Convert arrays into dataframe
        self.df = pd.DataFrame({"image_name" : image_names, "labels" : labels})
        self.df.to_csv("\Data\handwritten_hangul.csv")
        

    def _load_csv(self):
        """
        Private helper function to load already existing csvr
        """

    def _train_test_split(self):
        """
        Helper function to 
        """
    
    def load_image(self, image_name: str):
        """
        Helper function to load an image from file name and return as array
        """