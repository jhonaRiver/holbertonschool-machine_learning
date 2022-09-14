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
        activations (list): contains the activation functions used for each
                            layer of the network
        lambtha (float): L2 regularization parameter
        keep_prob (float): probability that a node will be kept for dropout
    Returns:
        keras model
    """
    inputs = K.Input(shape=(nx,))
    L2 = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            output = K.layers.Dense(layers[i], activation=activations[i],
                                    kernel_regularizer=L2)(inputs)
        else:
            dropout = K.layers.Dropout(1 - keep_prob)(output)
            output = K.layers.Dense(layers[i], activation=activations[i],
                                    kernel_regularizer=L2)(dropout)
    return K.models.Model(inputs=inputs, outputs=output)
