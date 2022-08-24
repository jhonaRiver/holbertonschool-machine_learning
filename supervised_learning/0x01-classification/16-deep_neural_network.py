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
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for layer in range(self.L):
            if type(layers[layer]) != int or layers[layer] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.weights['b' + str(layer + 1)] = np.zeros((layers[layer], 1))
            if layer == 0:
                He_et_al = np.random.randn(layers[layer], nx) * np.sqrt(2/nx)
                self.weights['W' + str(layer + 1)] = He_et_al
            else:
                He_et_al = np.random.randn(layers[layer], layers[layer - 1]) *\
                              np.sqrt(2/layers[layer-1])
                self.weights['W' + str(layer + 1)] = He_et_al
