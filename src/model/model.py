from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class SVCModel:
    def __init__(self):
        self.tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]
        self.cv = None
        self.model = None
        self.initialize()
    
    def initialize(self):
        self.cv = GridSearchCV(SVC(), self.tuned_parameters, refit=True,verbose=3)

    def train(self,x_train,y_train):
        self.cv.fit(x_train,y_train)
        self.model = self.cv.best_estimator_

    def predict_proba(self,X):
        return self.model.predict_proba(X), self.model.classes
    
    def predict(self,X):
        return self.model.predict(X)

class RandomForestModel:
    def __init__(self):
        self.model = None
        self.initialize()
    
    def initialize(self):
        self.model = RandomForestClassifier()
    
    def train(self,X,y):
        self.model.fit(X,y)
    
    def predict_proba(self,X):
        return self.model.predict_proba(X), self.model.classes_

    def predict(self,X):
        return self.model.predict(X), self.model.classes_
    
    

class Model:
    # This class provides an interface for the model (while this is not
    # strictly needed for a Random Forest classifier, it shows an example
    # of how the class could be constructed if the model is bespoke)
    def __init__(self,type) -> None:
        self.model = []
        self.initialize(type)
    
    def initialize(self,type):
        match type:
            case "SVC":
                self.model = SVCModel()
            case "RandomForest":
                self.model = RandomForestModel()
    
    def train(self,X,y):
        self.model.train(X,y)
        
    def predict_proba(self,X):
        prediction,classes = self.model.predict_proba(X)
        return [prediction, classes]

    def predict(self,X):
        prediction,classes = self.model.predict(X)
        return [prediction, classes]