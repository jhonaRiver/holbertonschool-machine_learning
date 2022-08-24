#!/usr/bin/env python3
"""
module DeepNeuralNetwork class
"""
import numpy as np


class DeepNeuralNetwork:
    """
    defines a deep neural network performing binary classification
    """
    def __init__(self, nx, layers):
        """
        class constructor
        Args:
            nx (int): number of input features
            layers (list): list representing the number of nodes in each layer
                           of the network

        Raises:
            TypeError: nx must be an integer
            ValueError: nx must be a positive integer
            TypeError: layers must be a list of positive integers
            TypeError: layers must be a list of positive integers
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for layer in range(self.L):
            if type(layers[layer]) != int or layers[layer] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights['b' + str(layer + 1)] = np.zeros((layers[layer], 1))
            if layer == 0:
                He_et_al = np.random.randn(layers[layer], nx) * np.sqrt(2/nx)
                self.__weights['W' + str(layer + 1)] = He_et_al
            else:
                He_et_al = np.random.randn(layers[layer], layers[layer - 1]) *\
                              np.sqrt(2/layers[layer-1])
                self.__weights['W' + str(layer + 1)] = He_et_al

    @property
    def L(self):
        """
        getter function
        Returns:
            int: number of layers in the neural network
        """
        return self.__L

    @property
    def cache(self):
        """
        getter function
        Returns:
            dict: dictionary to hold all intermediary values of the network
        """
        return self.__cache

    @property
    def weights(self):
        """
        getter function
        Returns:
            dict: dictionary to hold all weights and biased of the network
        """
        return self.__weights

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neural network
        Args:
            X (ndarray): contains the input data

        Returns:
            dict: output of the neural network and the cache
        """
        self.__cache["A0"] = X
        for layer in range(self.__L):
            weights = self.__weights
            cache = self.__cache
            fwp_A = np.matmul(weights["W" + str(layer + 1)], cache["A" +
                              str(layer)])
            fwp = fwp_A + weights["b" + str(layer + 1)]
            cache["A" + str(layer + 1)] = 1 / (1 + np.exp(-fwp))
        return cache["A" + str(self.__L)], cache

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
        cost = -1 / m * np.sum(Y * np.log(A) + (1-Y) * np.log(1.0000001-A))
        return cost
