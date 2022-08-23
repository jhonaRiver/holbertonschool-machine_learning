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
        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
