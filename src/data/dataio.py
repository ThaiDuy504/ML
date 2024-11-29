# Import libraries and classes required for this example:
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np


def to_dataframe(X,y):
    return pd.DataFrame(np.concatenate((X,y[:, np.newaxis]),axis=1))

def save(X,y,datapath):
    # Save the dataframe to a csv file
    df = to_dataframe(X,y)
    df.to_csv(datapath,header = False, index=False)

def load(datapath):
    df = pd.read_csv(datapath,header=None)
    return df
