# This cell sets up the MNIST dataset
import numpy as np
import pandas as pd
class MNIST:
    """
    sets up MNIST dataset from OpenML
    """

    def __init__(self):
        df = pd.read_csv("data/mnist_784.csv")

        # Create arrays for the features and the response variable
        # store for use later
        y = df['class'].values
        X = df.drop('class', axis=1).values

        # Convert the labels to numeric labels
        y = np.array(pd.to_numeric(y))
        # create training and validation sets
        self.train_x, self.train_y = X[:30000, :], y[:30000]
        self.val_x, self.val_y = X[5000:6000, :], y[5000:6000]

