#!/usr/bin/env python3
"""Script that creates a dataframe from a dictionary."""
import pandas as pd


# Define the dictionary with the data
data_dict = {'First': [0.0, 0.5, 1.0, 1.5],
             'Second': ['one', 'two', 'three', 'four']}

# Create the DataFrame with the specified row and column labels
df = pd.DataFrame(data_dict, index=['A', 'B', 'C', 'D'])
