import numpy as np
from scipy.special import expit
from random import randrange

#Computes the sigmoid of Z - WX + b - Numpy array
def sigmoid(Z):
    sigmoid = expit(Z)#np.exp(Z) / 1 + np.exp(Z)
    return sigmoid

#This function creates a vector of zeros of shape dimension,0 for w and initializes b to 0
def initializeParameters(dimension):
    w = np.zeros((dimension, 1))
    b = 0
    return w, b

#Cost function and its gradient
def propagate(w, b, X, Y, lam):
    m = X.shape[1]
    A = sigmoid(np.dot(w.transpose(),X)+b)

    cost = (-1/m) * (np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))) + (lam * (w**2))
    cost = cost[0]
    dw = (1.0/m) * np.dot(X,((A-Y).transpose()))
    db = (1.0/m) * np.sum(A-Y)
    grads = {"dw": dw,
             "db": db}

    return grads, cost

#Gradient descent
def optimize(w, b, X, Y, num_iterations, learning_rate, lam):

    costs = []
    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y, lam)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.transpose(),X)+b)

    for i in range(A.shape[1]):
        if A[0,i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, lam):

    w, b = initializeParameters(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, lam)
    print costs
    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w,b, X_test)
    Y_prediction_train = predict(w,b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))


    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations,
         "accuracy": format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100)}

    return d
