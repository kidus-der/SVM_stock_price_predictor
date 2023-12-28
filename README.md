# SVM_stock_price_predictor
Stock price prediction program that uses different SVM(Support Vector Model) models (specifically RBF, Linear, and Polynomial  regression models) to accurately predict the future stock price of a stock from past data.

Process:

Retrieve Data for Training

- Retrieve past stock price of a stock in a given time frame (I used 28 days) and collect that data into a csv file. I used Yahoo Finance for this, as you can set the timeframe you like and download a csv file straight from there.

Reshape the data

- use Numpy to reshape the array into a 2D array that we can then create regression models

Create and train our three models (Linear, Polynomial, and RBF Regression)

- I used the SVR method found in scikit.learn and then fit the models into our datasets


Plot results

- first generate a scatter plot of our data points
- plot our three regression models 
- label our axes

Get our predictions

- using the three helper functions I created (get_data, train_models, plot_results) we can then process our data to get predictions
using our final function 'process_data'

- we get our dates and prices array using get_data
- we train our models
- we plot our results

- we then reshape our date array into a 2D array (one row of samples and one column of features)
- we then use '.predict' to get our individual predictions, dependent on type of regression, using our reshaped input data
- we then return these predictions


