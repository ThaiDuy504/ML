import src.common.tools as tools
import os
import src.preprocess.preprocessImage as preprocessImage
from skimage.io import imread
import matplotlib.pyplot as plt


#load 1 image
def load_image(image_path):
    image = imread(image_path)
    return image

#preprocess 1 image
def preprocess_image(image):
    return preprocessImage.preprocessImg(image)

#predict 1 image
def predict_image(image,model):
    config = tools.load_config()
    modelpath = config["modelpath"] + model + ".p"
    print(modelpath)
    # assert(os.path.exists(modelpath))
    Model = tools.pickle_load(modelpath)
    [y_hat, classes] = Model.predict_proba([image])
    return y_hat[0], classes

def predict(img,model):
    _, img_flat = preprocess_image(img)
    y_hat, classes = predict_image(img_flat,model)
    return y_hat, classes
