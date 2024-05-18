import pandas as pd 
import numpy as np
import time
import datetime


class LogisticRegression:
    def __init__(self, n_features, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
         
    #Sigmoid method
    def _sigmoid(self, x):
        if x > 0:   
            z = np.exp(-x)
            return 1/(1+z)
        else:
            z = np.exp(x)
            return z/(1+z)

    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1-y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1 + y2)

    def feed_forward(self,X):
        z = np.dot(X, self.weights) + self.bias
        A = self._sigmoid(z)
        return A

    def fit_once(self, X, y):
        n_samples, n_features = X.shape
        
        # gradient descent
        for _ in range(self.n_iters):
            A = self.feed_forward(X)
            dz = A - y # derivative of sigmoid and bce X.T*(A-y)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(A - y)
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict_once(self, X):
        y_hat = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(y_hat)
        
        return np.array(y_predicted)
    