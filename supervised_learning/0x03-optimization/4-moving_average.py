#!/usr/bin/env python3
"""
module moving_average
"""
import numpy as np


def moving_average(data, beta):
    """
    calculates the weighted moving average of a data set
    Args:
        data (list): data to calculate the moving average of
        beta (float): weight used for the moving average
    Returns:
        list containing the moving averages of data
    """
    avg = []
    n = 0
    for i in range(len(data)):
        n = beta * n + (1 - beta) * data[i]
        avg.append(n / (1 - beta ** (i + 1)))
    return avg
