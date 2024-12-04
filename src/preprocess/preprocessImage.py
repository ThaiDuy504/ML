import src.common.tools as tools
import os
import numpy as np
import src.data.dataio as dataio
from skimage.io import imread
from skimage.transform import resize
# from skimage.color import rgb2gray

def preprocessImg(img):
    img_resized = resize(img,(150,150,3))
    img_flat = img_resized.flatten()
    img_flat = np.array(img_flat)
    return img_resized, img_flat

def preprocess(data):
    print("Preprocessing data..")
    config = tools.load_config()

    target = []
    flat_data = []
    # images = []
    DataDirectory = config[data]

    Categories = ["cats","dogs"]

    for i in Categories:
        print("Category is:",i,"\tLabel encoded as:",Categories.index(i))
        # Encode categories cute puppy as 0, icecream cone as 1 and red rose as 2
        target_class = Categories.index(i)
        # Create data path for all folders under MinorProject
        path = os.path.join(DataDirectory,i)
        # Image resizing, to ensure all images are of same dimensions
        for img in os.listdir(path):
            img_array = imread(os.path.join(path,img))
            # Skimage normalizes the value of image
            img_resized, img_flat = preprocessImg(img_array)
            flat_data.append(img_flat)
            # images.append(img_resized)
            target.append(target_class)
        # Convert list to numpy array format
    flat_data = np.array(flat_data)
    # images = np.array(images)
    target = np.array(target)


    #save processed data to csv file
    print("Saving processed data...")
    df = dataio.to_dataframe(flat_data,target)
    dataio.save(df, config["dataprocesseddirectory"] + data + ".p")
    print("Data saved")
    return df

if __name__ == "__main__":
    flat_data, target, images = preprocess("testingdata")
    print(flat_data.shape)
    print(target.shape)
    print(images.shape)
    print("Preprocessing done")