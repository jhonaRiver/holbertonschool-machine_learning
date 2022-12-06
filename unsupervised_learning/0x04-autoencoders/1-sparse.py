#!/usr/bin/env python3
"""Module autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Create a sparse autoencoder.

    Args:
        input_dims (int): contains the dimensions of the model input
        hidden_layers (list): contains the number of nodes for each hidden
                              layer in the encoder, for the decoder it should
                              be reversed
        latent_dims (int): contains the dimensions of the latent space
                           representation
        lambtha (float): regularization parameter used for L1 regularization
                         on the encoded output
    Returns:
        encoder: encoder model
        decoder: decoder model
        auto: sparse autoencoder model
    """
