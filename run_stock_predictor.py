from predictor import process_data

# define our stock data file and the date we want to predict
filename = 'stock_data/TSLA.csv'
prediction_date = [20231228]

# run our prediction function
linear_pred, poly_pred, rbf_pred = process_data(filename, prediction_date)

# print the predictions
print(f'Linear Model Prediction: {linear_pred}')
print(f'Polynomial Model Prediction: {poly_pred}')
print(f'RBF Model Prediction: {rbf_pred}')