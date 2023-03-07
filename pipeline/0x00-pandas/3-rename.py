#!/usr/bin/env python3
"""Rename script."""

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE
# Rename the Timestamp column to Datetime
df = df.rename(columns={'Timestamp': 'Datetime'})

# Convert the Datetime values to datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

# Display only the Datetime and Close columns
df = df[['Datetime', 'Close']]

print(df.tail())
