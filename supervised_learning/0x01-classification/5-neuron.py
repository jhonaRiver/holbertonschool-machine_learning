#!/usr/bin/env python3
"""
module for Neuron class
"""
import numpy as np


class Neuron:
    """
    defines a single neuron performing binary classification
    """
    def __init__(self, nx):
        """
        class constructor
        Args:
            nx (int): number of input features to the neuron

        Raises:
            TypeError: nx must be an integer
            ValueError: nx must be a positive integer
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        getter function
        Returns:
            float vector: weights vector for the neuron
        """
        return self.__W

    @property
    def b(self):
        """
        getter function
        Returns:
            int: bias for the neuron
        """
        return self.__b

    @property
    def A(self):
        """
        getter function
        Returns:
            int: activated output of the neuron (prediction)
        """
        return self.__A

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neuron
        Args:
            X (ndarray): contains the input data

        Returns:
            float: activated output of the neuron
        """
        fwp = np.matmul(self.__W, X) + self.__b
        self.__A = 1/(1+np.exp(-fwp))
        return self.__A

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
        total_cost = -(1/m) * np.sum(np.multiply(Y, np.log(A)) +
                                     np.multiply(1-Y, np.log(1.0000001-A)))
        return total_cost

    def evaluate(self, X, Y):
        """
        evaluates the neuron's predictions
        Args:
            X (ndarray): contains the input data
            Y (ndarray): contains the correct labels for the input data

        Returns:
            ndarray: neuron's prediction and the cost of the network
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        prediction = np.where(self.__A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        calculates one pass of gradient descent on the neuron
        Args:
            X (ndarray): contains the input data
            Y (ndarray): contains the correct labels for the input data
            A (ndarray): contains the actived output of the neuron
            alpha (float, optional): learning rate. Defaults to 0.05.
        """
        m = Y.shape[1]
        dW = np.matmul(A-Y, X.T) / m
        db = np.sum(A-Y) / m
        self.__W = self.__W - (dW * alpha)
        self.__b = self.__b - (db * alpha)
