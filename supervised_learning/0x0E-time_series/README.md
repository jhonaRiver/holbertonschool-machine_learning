# 0x0E Time Series Forecasting

> Time-series forecasting is a technique that utilizes historical and current data to predict future values over a period of time or a specific point in the future. By analyzing data that we stored in the past, we can make informed decisions that can guide our business strategy and help us understand future trends

At the end of this project I was able to answer these conceptual questions:

* What is time series forecasting?
* What is a stationary process?
* What is a sliding window?
* How to preprocess time series data
* How to create a data pipeline in tensorflow for time series data
* How to perform time series forecasting with RNNs in tensorflow

## Tasks

0. Bitcoin (BTC) became a trending topic after its [price](https://www.google.com/search?q=bitcoin+price) peaked in 2018. Many have sought to predict its value in order to accrue wealth. Let’s attempt to use our knowledge of RNNs to attempt just that.

    Given the [coinbase](https://drive.google.com/file/d/16MgiuBfQKzXPoWFWi2w-LKJuZ7LgivpE/view) and [bitstamp](https://drive.google.com/file/d/15A-rLSrfZ0td7muSrYHy0WX9ZqrMweES/view) datasets, write a script, `forecast_btc.py`, that creates, trains, and validates a keras model for the forecasting of BTC:

    * Your model should use the past 24 hours of BTC data to predict the value of BTC at the close of the following hour (approximately how long the average transaction takes):
    * The datasets are formatted such that every row represents a 60 second time window containing:
        * The start time of the time window in Unix time
        * The open price in USD at the start of the time window
        * The high price in USD within the time window
        * The low price in USD within the time window
        * The close price in USD at end of the time window
        * The amount of BTC transacted in the time window
        * The amount of Currency (USD) transacted in the time window
        * The [volume-weighted average price](https://en.wikipedia.org/wiki/Volume-weighted_average_price#:~:text=In%20finance%2C%20volume%2Dweighted%20average,traded%20over%20the%20trading%20horizon.) in USD for the time window
    * Your model should use an RNN architecture of your choosing
    * Your model should use mean-squared error (MSE) as its cost function
    * You should use a `tf.data.Dataset` to feed data to your model

    Because the dataset is [raw](https://en.wikipedia.org/wiki/Raw_data), you will need to create a script, `preprocess_data.py` to preprocess this data. Here are some things to consider:

    * Are all of the data points useful?
    * Are all of the data features useful?
    * Should you rescale the data?
    * Is the current time window relevant?
    * How should you save this preprocessed data?

## Results

| Filename |
| ------ |
| [forecast_btc.py]()|
| [preprocess_data.py]()|
