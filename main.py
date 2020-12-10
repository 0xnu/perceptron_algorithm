from perceptron_algorithm import PerceptronAlgorithm
from data import Data
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    pass

    df = Data()
    
    X_train, X_test, Y_train, Y_test, X, Y = df.preprocessing(df.dataset)
    
    perceptron = PerceptronAlgorithm()
    
    df.plot(df.dataset)        
       
      
    # epochs = 10000 and lr = 0.3
    wt_matrix = perceptron.fit(X_train, Y_train, 10000, 0.3)
    
    # Predictions
    Y_pred_test = perceptron.predict(X_test)
    
    # Accuracy score
    print(accuracy_score(Y_pred_test, Y_test))
