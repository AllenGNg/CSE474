import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import matplotlib.pyplot as plt
import random

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    # your code here
    return 1.0 / (1.0 + np.exp(-z))

def derivative_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))
"""
def preprocess(): # preprocess_small() from Piazza, added to use a smaller test instead of the original
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_sample.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     - feature selection


    mat = loadmat('mnist_sample.mat')
        # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(4996, 784))
    validation_preprocess = np.zeros(shape=(1000, 784))
    test_preprocess = np.zeros(shape=(996, 784))
    train_label_preprocess = np.zeros(shape=(4996,))
    validation_label_preprocess = np.zeros(shape=(1000,))
    test_label_preprocess = np.zeros(shape=(996,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 100  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[100:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 100] = tup[tup_perm[0:100], :]
            validation_len += 100

            validation_label_preprocess[validation_label_len:validation_label_len + 100] = label
            validation_label_len += 100

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    # Feature selection is a process where your automatically select
    # those Features in your data that contribute most to the prediction value
    # in which you are most interested.

    # Get rid of blank space around numbers, get rid of too thin and too fat numbers,
    # check ifthere are pixels not on number.

    i = 0
    done = len(train_data.T)
    while(i < done):
        delete_me = True
        for j in ((train_data.T)[i]):
            if j != 0:
                delete_me = False
                break;
        if delete_me:
            train_data =np.delete(train_data,i,1)
            test_data = np.delete(test_data,i,1)
            done = done -1
            delete_me = True
        else:
            i = i + 1

    return train_data, train_label, validation_data, validation_label, test_data, test_label
"""

def preprocess():
    """#Input:
    #Although this function doesn't have any input, you are required to load
    the MNIST data set from file 'mnist_all.mat'.
    Output:
    train_data: matrix of training set. Each row of train_data contains
        feature vector of a image
    train_label: vector of label corresponding to each image in the training
        set
    validation_data: matrix of training set. Each row of validation_data
        contains feature vector of a image
    validation_label: vector of label corresponding to each image in the
        training set
    test_data: matrix of training set. Each row of test_data contains
        feature vector of a image
    test_label: vector of label corresponding to each image in the testing
        set

    Some suggestions for preprocessing step:
        - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data
    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, the training data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    obj_grad = np.array([])

    """********FORWARD PASS**********"""
    x1 = np.hstack([np.ones([training_data.shape[0],1]),training_data])
    hidden_layer = sigmoid(np.dot(x1,w1.T))
    x2 = np.hstack([np.ones([hidden_layer.shape[0],1]),hidden_layer])
    output_layer = sigmoid(np.dot(x2,w2.T))

    """*********** BACKWARD PROPOGATESTUFF**********"""
    y = np.zeros((len(training_data),n_class))
    for i in range(0,len(training_label)):
        val = int(training_label[i])
        y[i][val] = 1

    error = y*np.log(output_layer)+(1-y)*np.log(1-output_layer)
    obj_val = - np.sum(error)/len(training_data)
    error = ((np.sum(w1**2) + np.sum(w2**2))/(2*len(training_data)))*lambdaval
    obj_val = obj_val + error

    delta_out = (output_layer - y)
    delta_hidden = np.dot(delta_out, w2)*(x2*(1-x2))
    new_w1 = np.dot(delta_hidden.T, x1)
    new_w2 = np.dot(delta_out.T,x2)

    new_w1 = new_w1[:-1:]

    grad_w1 =  new_w1
    grad_w2 =  new_w2

    grad_w1 = (grad_w1 + lambdaval*w1)/len(training_data.T)
    grad_w2 = (grad_w2 + lambdaval*w2)/len(training_data.T)
    obj_grad = np.concatenate((grad_w1.flatten(),grad_w2.flatten()),0)

    return (obj_val, obj_grad)

def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    labels = np.array([])

    return labels

"""**************Neural Network Script Starts here********************************"""

n_input = 5
n_hidden = 3
n_class = 2
training_data = np.array([np.linspace(0,1,num=5),np.linspace(1,0,num=5)])
training_label = np.array([0,1])
lambdaval = 0
params = np.linspace (-5,5,num=26)
#args = (n_input, n_hidden, n_class, training_data, training_label, lambdaval)

one,two = nnObjFunction(params,n_input,n_hidden,n_class,training_data,training_label,lambdaval)
print(one)
print(two)
