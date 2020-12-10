import sklearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class Data :

    
    def __init__(self):
        # Load the dataset
        # Reference: https://scikit-learn.org/stable/datasets/index.html
        dataset = sklearn.datasets.load_wine()
        self.dataset = dataset
        
        
    def preprocessing (self, dataset):
        
        # Conversion to dataframe
        data = pd.DataFrame(dataset.data, columns = dataset.feature_names)
        data["class"] = dataset.target       
        
        # Standardisation MinMaxScaler
        X = data.drop("class", axis = 1)
        Y = data["class"]
        mnscaler = MinMaxScaler()
        X = mnscaler.fit_transform(X)
        X = pd.DataFrame(X, columns=data.drop("class",axis = 1).columns)
        
        # Train test split.
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, stratify = Y, random_state = 1)
        
        return X_train.values, X_test.values, Y_train, Y_test, X, Y
    
    
    
    def plot (self,dataset):
        
        # Conversion to dataframe
        data = pd.DataFrame(dataset.data, columns = dataset.feature_names)
        data["class"] = dataset.target    
        
        data['class'].value_counts().plot(kind = "barh")
        plt.xlabel("Count")
        plt.ylabel("Classes")
        plt.show()
