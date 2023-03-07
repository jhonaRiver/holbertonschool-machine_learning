#!/usr/bin/env python3
"""Module from_numpy."""
import pandas as pd


def from_numpy(array):
    """
    Create a dataframe from a ndarray.

    Args:
        array (ndarray): from which you should create the dataframe
    Returns:
        newly created dataframe
    """
    columns = list(map(chr, range(65, 91)))[:array.shape[1]]
    df = pd.DataFrame(data=array, columns=columns)
    return df
