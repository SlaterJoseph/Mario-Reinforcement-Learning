import numpy as np

class DataGenerator:
    """
    Normalizing the images took to much RAM, so a data generator was created to help alleviate that issue
    """


    def normalize_data(self, state: np.array) -> np.array:
