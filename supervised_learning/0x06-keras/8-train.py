#!/usr/bin/env python3
"""
module train_model
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
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
        learning_rate_decay (boolean, optional): indicates whether learning
                                                 rate decay should be used.
                                                 Defaults to False.
        alpha (float, optional): initial learning rate. Defaults to 0.1
        decay_rate (int, optional): decay rate. Defaults to 1.
        save_best (boolean, optional): indicates whether to save the model
                                       after each epoch if it is the best.
                                       Defaults to False.
        filepath (str, optional): where the model should be saved. Defaults to
                                  None.
        verbose (boolean, optional): determines if output should be printed
                                     during training. Defaults to True.
        shuffle (boolean, optional): determines whether to shuffle the batches
                                     every epoch. Defaults to False.
    Returns:
        History object generated after training the model
    """
    early_stopping_callback = []

    def scheduler(epoch):
        """
        gets learning rate of each epoch
        Args:
            epoch (int): current epoch
        Returns:
            learning rate
        """
        return alpha / (1 + decay_rate * epoch)
    if validation_data and early_stopping:
        early_stopping_callback.append(
                K.callbacks.EarlyStopping(patience=patience))
    if validation_data and learning_rate_decay:
        learning_rate_decay_callback = K.callbacks.LearningRateScheduler(
                scheduler, verbose=1)
        early_stopping_callback.append(learning_rate_decay_callback)
    if save_best:
        checkpoint = K.callbacks.ModelCheckpoint(filepath, save_best_only=True)
        early_stopping_callback.append(checkpoint)
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       callbacks=early_stopping_callback,
                       validation_data=validation_data, verbose=verbose,
                       shuffle=shuffle)
