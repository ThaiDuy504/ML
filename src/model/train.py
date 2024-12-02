import os
from src.model import model
import src.common.tools as tools
import src.data.dataio as dataio
import src.preprocess.preprocessImage as preprocessImage
from sklearnex import patch_sklearn, config_context
patch_sklearn()

def train(config):
    # Load the data
    filepath = config["dataprocesseddirectory"] + "trainingdata.csv"

    print("loading data....")
    if(not os.path.exists(filepath)):
        preprocessImage.preprocess("trainingdata")

    df = dataio.load(filepath)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    # Train the model
    print("trainnning....")
    Model = model.Model(config["model"])
    with config_context(target_offload="gpu:0"):
        Model.train(X,y)

    # Save the trained model
    print("saving model....")
    tools.pickle_dump(config["modelpath"] + config["model"] + ".p", Model)
    print("saving model successful")
if __name__ == "__main__":
    config = tools.load_config()
    train(config)