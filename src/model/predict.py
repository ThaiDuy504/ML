# from src.model import model
import os
from src.evaluate import evaluate
import src.common.tools as tools
import src.data.dataio as dataio
import src.preprocess.preprocessImage as preprocessImage

def predict(config):
    # Load the data to make predictions on
    print("Loading testing data")
    filepath = config["dataprocesseddirectory"] + "testingdata.p"
    if(not os.path.exists(filepath)):
        df = preprocessImage.preprocess("testingdata")
    else:
        df = dataio.load(filepath)

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    # Load the model
    modelpath = config["modelpath"] + config["model"] + ".p"
    Model = tools.pickle_load(modelpath)
    
    # Make predictions from the trained model
    [y_hat, classes] = Model.predict(X)
    
    # Save results to a convenient data structure
    Result = evaluate.Results(y,y_hat,classes)
    resultspath = config["resultsrawpath"] + config["model"] + ".p"
    tools.pickle_dump(resultspath,Result)

if __name__ == "__main__":
    config = tools.load_config()
    predict(config)