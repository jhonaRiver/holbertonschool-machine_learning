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
    x = K.layers.Dense(layers[0], activation=activations[0],
                       kernel_regularizer=K.regularizers.l2(lambtha),
                       input_shape=(nx,))(inputs)
    y = x
    for i in range(1, len(layers)):
        if i == 1:
            x = K.layers.Dropout(1 - keep_prob)(x)
        else:
            y = K.layers.Dropout(1 - keep_prob)(y)
        y = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_regularizer=K.regularizers.l2(lambtha))(y)
    model = K.Model(inputs=inputs, outputs=y)
    return model
