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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        calculates one pass of gradient descent on the neural network
        Args:
            X (ndarray): contains the input data
            Y (ndarray): contains the correct labels for the input data
            A1 (int): output of the hidden layer
            A2 (int): predicted output
            alpha (float, optional): learning rate. Defaults to 0.05.
        """
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = 1/m * np.matmul(dz2, A1.T)
        db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.matmul(self.__W2.T, dz2) * A1 * (1 - A1)
        dw1 = 1/m * np.matmul(dz1, X.T)
        db1 = 1/m * np.sum(dz1, axis=1, keepdims=True)
        self.__W1 = self.__W1 - alpha * dw1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dw2
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        trains the neural network
        Args:
            X (ndarray): contains the input data
            Y (ndarray): contains the correct labels for the input data
            iterations (int, optional): number of iterations to train over.
                                        Defaults to 5000.
            alpha (float, optional): learning rate. Defaults to 0.05.

        Raises:
            TypeError: iterations must be an integer
            ValueError: iterations must be a positive integer
            TypeError: alpha must be a float
            ValueError: alpha must be positive

        Returns:
            ndarray: evaluation of the training data after iterations of
                     training have occurred
        """
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        for training in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        return self.evaluate(X, Y)
