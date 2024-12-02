from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# MLP model
from sklearn.neural_network import MLPClassifier
# KNN model
from sklearn.neighbors import KNeighborsClassifier
# Decision Tree model
from sklearn.tree import DecisionTreeClassifier
# Logistic Regression model
from sklearn.linear_model import LogisticRegression

#stacked model random forest, decision tree, knn, mlp, svc
from sklearn.ensemble import StackingClassifier

class StackedModel:
    def __init__(self):
        self.model = None
        self.initialize()
    
    def initialize(self):
        self.model = StackingClassifier(estimators=[('rf', RandomForestClassifier()), ('dt', DecisionTreeClassifier()), ('knn', KNeighborsClassifier()), ('mlp', MLPClassifier()), ('svc', SVC())], final_estimator=LogisticRegression())
    
    def train(self,x_train,y_train):
        self.model.fit(x_train,y_train)

    def predict_proba(self,X):
        return self.model.predict_proba(X), self.model.classes
    
    def predict(self,X):
        return self.model.predict(X), self.model.classes_

class LogisticRegressionModel:
    def __init__(self):
        self.model = None
        self.initialize()
    
    def initialize(self):
        self.model = LogisticRegression()
    
    def train(self,X,y):
        self.model.fit(X,y)
    
    def predict_proba(self,X):
        return self.model.predict_proba(X), self.model.classes_

    def predict(self,X):
        return self.model.predict(X), self.model.classes_
    
class DecisionTreeModel:
    def __init__(self):
        self.model = None
        self.initialize()
    
    def initialize(self):
        self.model = DecisionTreeClassifier()
    
    def train(self,X,y):
        self.model.fit(X,y)
    
    def predict_proba(self,X):
        return self.model.predict_proba(X), self.model.classes_

    def predict(self,X):
        return self.model.predict(X), self.model.classes_


class KNNModel:
    def __init__(self):
        self.tuned_parameters = [{'n_neighbors': [2, 3, 5, 7, 11, 19], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}]
        self.cv = None
        self.model = None
        self.initialize()
    
    def initialize(self):
        self.cv = GridSearchCV(KNeighborsClassifier(), self.tuned_parameters, refit=True,verbose=3)

    def train(self,x_train,y_train):
        self.cv.fit(x_train,y_train)
        self.model = self.cv.best_estimator_

    def predict_proba(self,X):
        return self.model.predict_proba(X), self.model.classes
    
    def predict(self,X):
        return self.model.predict(X), self.model.classes_
    
class MLPModel:
    def __init__(self):
        # self.tuned_parameters = [{'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)], 'activation': ['tanh', 'relu'],'solver': ['sgd', 'adam'],'alpha': [0.0001, 0.05],'learning_rate': ['constant','adaptive']}]
        self.cv = None
        self.model = None
        self.initialize()
    
    def initialize(self):
        self.model = MLPClassifier(hidden_layer_sizes=(512,256,128,), max_iter=1000,activation='tanh',solver='sgd',random_state=42,early_stopping=True,learning_rate='adaptive',alpha=0.0001)

    def train(self,x_train,y_train):
        self.model.fit(x_train,y_train)

    def predict_proba(self,X):
        return self.model.predict_proba(X), self.model.classes
    
    def predict(self,X):
        return self.model.predict(X), self.model.classes_

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
        return self.model.predict(X), self.model.classes_

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
            case "LogisticRegression":
                self.model = LogisticRegressionModel()
            case "DecisionTree":
                self.model = DecisionTreeModel()
            case "KNN":
                self.model = KNNModel()
            case "MLP":
                self.model = MLPModel()
            case "Stacked":
                self.model = StackedModel()
        
    
    def train(self,X,y):
        self.model.train(X,y)
        
    def predict_proba(self,X):
        prediction,classes = self.model.predict_proba(X)
        return [prediction, classes]

    def predict(self,X):
        prediction,classes = self.model.predict(X) 
        return [prediction, classes]