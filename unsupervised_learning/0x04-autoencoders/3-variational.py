#!/usr/bin/env python3
"""Module autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Create a variational autoencoder.

    Args:
        input_dims (int): contains the dimensions of the model input
        hidden_layers (list): contains the number of nodes for each hidden
                              layer in the encoder, for the decoder they
                              should be reversed
        latent_dims (int): contains the dimensions of the latent space
                           representation
    Returns:
        encoder: encoder model, which should output the latent representation,
                 the mean, and the log variance
        decoder: decoder model
        auto: full autoencoder model
    """
