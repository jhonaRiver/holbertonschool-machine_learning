#!/usr/bin/env python3
"""Module autoencoder."""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Create a convolutional autoencoder.

    Args:
        input_dims (tuple): contains the dimensions of the model input
        filters (list): contains the number of filters for each convolutional
                        layer in the encoder, for the decoder they should be
                        reversed
        latent_dims (tuple): contains the dimensions of the latent space
                             representation
    Returns:
        encoder: encoder model
        decoder: decoder model
        auto: full autoencoder model
    """
