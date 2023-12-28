# SVM_stock_price_predictor
Stock price prediction program that uses different SVM(Support Vector Model) models (specifically RBF, Linear, and Polynomial  regression models) to accurately predict the future stock price of a stock from past data.

Process:

Retrieve Data for Training

- Retrieve past stock price of a stock in a given time frame (I used 28 days) and collect that data into a csv file. I used Yahoo Finance for this, as you can set the timeframe you like and download a csv file straight from there.

Reshape the data

- use Numpy to reshape the array into a 2D array that we can then create regression models



