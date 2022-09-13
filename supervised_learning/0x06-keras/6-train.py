#!/usr/bin/env python3
"""
module train_model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    trains a model using mini-batch gradient descent
    Args:
        network (keras): model to train
        data (ndarray): contains the input data
        labels (ndarray): contains the labels of data
        batch_size (int): size of the batch used for mini-batch gradient
                          descent
        epochs (int): number of passes through data for mini-batch gradient
                      descent
        validation_data (ndarray, optional): data to validate the model with.
                                             Defaults to None.
        early_stopping (boolean, optional): indicates whether early stopping
                                            should be used. Defaults to False.
        patience (int, optional): patience used for early stopping. Defaults
                                  to 0.
        verbose (boolean, optional): determines if output should be printed
                                     during training. Defaults to True.
        shuffle (boolean, optional): determines whether to shuffle the batches
                                     every epoch. Defaults to False.
    Returns:
        History object generated after training the model
    """
    stopping = []
    if early_stopping and validation_data:
        stopping.append(K.callbacks.EarlyStopping(patience=patience))
    iteration = network.fit(data, labels, batch_size=batch_size,
                            epochs=epochs, verbose=verbose, shuffle=shuffle,
                            validation_data=validation_data,
                            callbacks=stopping)
    return iteration
