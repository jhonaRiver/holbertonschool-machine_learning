#!/usr/bin/env python3
"""
module early_stopping
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    determines if you should stop gradient descent early
    Args:
        cost (float): current validation cost of the neural network
        opt_cost (float): lowest recorded validation cost of the neural
                          network
        threshold (float): threshold used for early stopping
        patience (int): patience count used for early stopping
        count (int): count of how long the threshold has not been met
    Returns:
        boolean of whether the network should be stopped early, followed by
        the updated count
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    return count >= patience, count
