#!/usr/bin/env python3
"""Visualize script."""

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove Weighted_Price column
df.drop(columns=['Weighted_Price'], inplace=True)

# Rename Timestamp column to Date
df.rename(columns={'Timestamp': 'Date'}, inplace=True)

# Convert timestamp to date
df['Date'] = pd.to_datetime(df['Date'], unit='s').dt.to_period('d')
df = df.loc[df['Date'] >= "2017-01-01"]
# Index dataframe on Date
df.set_index('Date', inplace=True)

# Forward fill missing Close values
df['Close'].fillna(method='ffill', inplace=True)

# Fill missing values in other columns
df['High'].fillna(value=df['Close'], inplace=True)
df['Low'].fillna(value=df['Close'], inplace=True)
df['Open'].fillna(value=df['Close'], inplace=True)
df[['Volume_(BTC)', 'Volume_(Currency)']] = df[[
    'Volume_(BTC)', 'Volume_(Currency)']].fillna(0)

# Resample data by day and aggregate values
df = df.resample('D').agg({'High': 'max', 'Low': 'min', 'Open': 'mean',
                           'Close': 'mean', 'Volume_(BTC)': 'sum',
                           'Volume_(Currency)': 'sum'})

# Plot the data
df.plot(kind='line', subplots=True, layout=(
    3, 2), figsize=(15, 15), sharex=True)
plt.show()
