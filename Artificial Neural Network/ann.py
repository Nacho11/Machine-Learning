import numpy as np
from scipy.special import expit

#Computes the sigmoid of Z - WX + b - Numpy array
def sigmoid(Z):
    sigmoid = expit(Z)#np.exp(Z) / 1 + np.exp(Z)
    return sigmoid

# Returns the size of the input layer, hidden layer and output layer
def layerSizes(input_data_set, labels, hidden_layer_size):
    size_of_input_layer = input_data_set.shape[0] # Number of input units
    size_of_output_layer = labels.shape[0] #Number output units
    size_of_hidden_layer = hidden_layer_size #Number of hidden layers
    return (size_of_input_layer, size_of_hidden_layer, size_of_output_layer)

# Initializes the parameters W1, b1 - for Layer 1(hidden layer) and W2, b2 for Layer 2(output layer)
def initializeParameters(size_of_input_layer, size_of_hidden_layer, size_of_output_layer):
    W1 = np.random.uniform(-0.1, 0.1, (size_of_hidden_layer, size_of_input_layer)) # Matrix containing all the initial random weights going to hidden layer
    b1 = np.zeros((size_of_hidden_layer, 1)) #Vector containing all the bias values going to hidden layer
    W2 = np.random.uniform(-0.1, 0.1, (size_of_output_layer, size_of_hidden_layer)) # Matrix containing all the initial random weights going to output layer
    b2 = np.zeros((size_of_output_layer, 1)) #Vector containing all the bias values going to output layer
    parameters = {  # Dictionary containing all the initialized parameters
    'W1':W1,
    'W2':W2,
    'b1':b1,
    'b2':b2
    }
    return parameters

# Forward propagates the network
def forwardPropagation(input_data, parameters):
    #print parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.add(np.dot(W1, input_data), b1) #Value of Z1 - linear function which is WX + b Layer 1
    A1 = sigmoid(Z1) #Value of A1 - activation function which is sigmoid Layer 1
    Z2 = np.add(np.dot(W2, A1), b2) #Value of Z2 - linear function which is WX + b Layer 2
    A2 = sigmoid(Z2) #Value of A2 - activation function which is sigmoid Layer 2

    cache = {   # caching all the forward propagation values into cache
    "Z1": Z1,
    "A1": A1,
    "Z2": Z2,
    "A2": A2
    }
    return cache

# Computes the cost give the predicted value from the last layer
def computeCost(A2, labels, parameters, weight_decay):
    '''
    Using the squared loss function to compute the cost
    forumla - 1/2 * summation(y[i] - A2[i])
    '''
    #print labels
    #print A2.shape
    #print labels.shape
    W1 = parameters["W1"]
    #print np.sum(W1)
    W2 = parameters["W2"]
    #print W2
    cost = (-1/2.0 * np.sum([labels] - A2)) + weight_decay * ((np.sum(W1)+np.sum(W2))**2)
    cost = np.squeeze(cost)
    return cost

#Returns the gradients with respect to the parameters
def backwardPropagation(input_data, labels, parameters, cache):
    sample_size = 1
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - labels
    dW2 = 1.0/sample_size * np.dot(dZ2, A1.transpose())
    db2 = 1.0/sample_size * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.transpose(),dZ2), A1*(1-A1))
    #print dZ1
    #print input_data
    dW1 = 1.0/sample_size * np.dot(dZ1, W1)
    db1 = 1.0/sample_size * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
    "dW1" : dW1,
    "dW2" : dW2,
    "db1" : db1,
    "db2" : db2
    }

    return gradients

#Returns the updated parameters
def updateParameters(parameters, gradients, learning_rate):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
    'W1': W1,
    'b1': b1,
    'W2': W2,
    'b2': b2
    }

    return parameters

def ann_model(input_data, labels, size_of_hidden_layer, num_of_iterations, weight_decay):
    size_of_input_layer = layerSizes(input_data, labels, size_of_hidden_layer)[0]
    size_of_output_layer = layerSizes(input_data, labels, size_of_hidden_layer)[2]
    #print labels
    parameters = initializeParameters(size_of_input_layer, size_of_hidden_layer, size_of_output_layer)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_of_iterations):
        #print input_data.shape
        for j in range(0, input_data.shape[1]):
            #print input_data[:, j]
            #print input_data[:, j].shape
            cache = forwardPropagation(input_data[:, j], parameters)
            #print labels
            #print cache["A2"]
            cost = computeCost(cache["A2"], labels[:, j], parameters, weight_decay)

            gradients = backwardPropagation(np.array([input_data[:, j]]), labels[:, j], parameters, cache)

            parameters = updateParameters(parameters, gradients, 0.01)

        print "Cost after iteration %i = %f" %(i, cost)

    return parameters


def predict(parameters, test_set):
    cache = forwardPropagation(test_set, parameters)
    A2 = cache["A2"]
    #print A2
    predictions = A2 < 0.5
    return predictions
