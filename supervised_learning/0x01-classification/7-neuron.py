#!/usr/bin/env python3
"""
module for Neuron class
"""
import numpy as np
import matplotlib.pyplot as plt


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

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        trains the neuron
        Args:
            X (ndarray): contains input data
            Y (ndarray): contains the correct labels for the input data
            iterations (int, optional): number of iterations to train over.
                                        Defaults to 5000.
            alpha (float, optional): learning rate. Defaults to 0.05.
            verbose (bool, optional): defines whether or not to print
                                      information about the training.
                                      Defaults to True.
            graph (bool, optional): defines whether or not to graph
                                    information about the training once it has
                                    completed. Defaults to True.
            step (int, optional): steps of the graph. Defaults to 100.

        Raises:
            TypeError: iterations must be an integer
            ValueError: iterations must be a positive integer
            TypeError: alpha must be a float
            ValueError: alpha must be positive
            TypeError: step must be an integer
            ValueError: step must be positive and <= iterations

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
        if verbose or graph:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        Cost = []
        Iterations = []
        for training in range(iterations):
            a, cost = self.evaluate(X, Y)
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            if (training % step == 0):
                Cost.append(cost)
                Iterations.append(training)
                if verbose:
                    print("Cost after {} iterations: {}".format(training,
                                                                cost))
        if graph:
            plt.plot(Iterations, Cost, 'b')
            plt.xlabel('iteration')
            plt.xlim(0, 3000)
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return self.evaluate(X, Y)
