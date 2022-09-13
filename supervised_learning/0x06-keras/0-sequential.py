#!/usr/bin/env python3
"""
module build_model
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network with the Keras library
    Args:
        nx (int): number of input features to the network
        layers (list): contains the number of nodes in each layer of the
                       network
        activations (list): contains the activation functions for each layer
                            of the network
        lambtha (float): L2 regularization parameter
        keep_prob (float): probability that a node will be kept for dropout
    Returns:
        keras model
    """
    model = K.Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[0], activation=activations[0],
                                     input_shape=(nx,), kernel_initializer=K.
                                     initializers.he_normal(),
                                     kernel_regularizer=K.
                                     regularizers.l2(lambtha)))
        else:
            model.add(K.layers.Dropout(1 - keep_prob))
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                      kernel_regularizer=K.regularizers.l2(lambtha)))
    return model
