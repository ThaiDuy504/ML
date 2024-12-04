# Import libraries and classes required for this example:
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np

def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if (
                c_min > np.finfo(np.float32).min
                and c_max < np.finfo(np.float32).max
            ):
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

def to_dataframe(X,y):
    df =  pd.DataFrame(np.concatenate((X,y[:, np.newaxis]),axis=1))
    df.info(verbose=True)
    print("optimizing dataframe")
    df = reduce_mem_usage(df)
    print("done optimizing")
    return df

def save(df,datapath):
    # Save the dataframe to a csv file
    df.to_pickle(datapath)

def load(datapath):
    df = pd.read_pickle(datapath)
    return df
