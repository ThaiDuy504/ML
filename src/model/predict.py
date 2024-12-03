# from src.model import model
import os
from src.evaluate import evaluate
import src.common.tools as tools
import src.data.dataio as dataio
import src.preprocess.preprocessImage as preprocessImage
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def draw_chart(y_true, y_pred, classes):
    # Draw a chart of the predictions from y_true and y_pred
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.imshow(confusion_matrix(y_true,y_pred), cmap='Blues')
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j,i,confusion_matrix(y_true,y_pred)[i,j],ha='center',va='center',color='red')

    plt.show()

def predict(config):
    # Load the data to make predictions on
    filepath = config["dataprocesseddirectory"] + "testingdata.csv"
    if(not os.path.exists(filepath)):
        df,_ = preprocessImage.preprocess("testingdata")
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