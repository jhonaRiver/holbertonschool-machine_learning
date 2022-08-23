#!/usr/bin/env python3
"""
module for NeuralNetwork class
"""
import numpy as np


class NeuralNetwork:
    """
    defines a neural network with one hidden layer performing binary
    classification
    """
    def __init__(self, nx, nodes):
        """
        class constructor
        Args:
            nx (int): number of input features
            nodes (int): number of nodes found in the hidden layer

        Raises:
            TypeError: nx must be an integer
            ValueError: nx must be a positive integer
            TypeError: nodes must be an integer
            ValueError: nodes must be a positive integer
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        getter function
        Returns:
            ndarray: weights vector for the hidden layer
        """
        return self.__W1

    @property
    def b1(self):
        """
        getter function
        Returns:
            ndarray: bias for the hidden layer
        """
        return self.__b1

    @property
    def A1(self):
        """
        getter function
        Returns:
            int: activated output for the hidden layer
        """
        return self.__A1

    @property
    def W2(self):
        """
        getter function
        Returns:
            ndarray: weights vector for the output neuron
        """
        return self.__W2

    @property
    def b2(self):
        """
        getter function
        Returns:
            int: bias for the output neuron
        """
        return self.__b2

    @property
    def A2(self):
        """
        getter function
        Returns:
            int: activated output for the output neuron (prediction)
        """
        return self.__A2

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neural network
        Args:
            X (ndarray): contains the input data

        Returns:
            int: predictions
        """
        fwp = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1/(1+np.exp(-fwp))
        self.__A2 = 1 / (1 + np.exp(-(np.matmul(self.__W2, self.__A1) +
                                      self.__b2)))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        calculates the cost of the model using logistic regression
        Args:
            Y (ndarray): contains the correct labels for the input data
            A (ndarray): contains the activated output of the neuron

        Returns:
            float: cost
        """
        m = Y.shape[1]
        cost = -1/m * np.sum(Y * np.log(A) + (1-Y) * (np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        """
        evaluates the neural network's predictions
        Args:
            X (ndarray): contains the input data
            Y (ndarray): contains the correct labels for the input data

        Returns:
            ndarray, float: neuron's prediction and the cost of the network
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        return prediction, cost
