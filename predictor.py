import csv
import numpy as np
import sklearn.svm as sk
import matplotlib.pyplot as plt
from sklearn.svm import SVR

def get_data(filename):

    dates, prices = [], [] #initialize empty arrays for dates and prices

    # open csv file, read, and append data
    # to arrays
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader) #skip first row (containing column descriptors)
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0])) #split the date string into int values, and append date to array
            prices.append(float(row[1])) #append opening price to array

    return dates, prices

def train_models(dates, prices):

    dates = np.reshape(dates, (len(dates), 1)) #reshape dates array to 2D array

    # create our three models used to predict
    linear_model = SVR(kernel = 'linear', C = 1e3)
    polynomial_model = SVR(kernel = 'poly', C = 1e3, degree = 2) 
    rbf_model = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1)

    # train our models using the '.fit' method using our two arrays for training
    linear_model.fit(dates, prices)
    polynomial_model.fit(dates, prices)
    rbf_model.fit(dates, prices)

    return linear_model, polynomial_model, rbf_model

def plot_results(dates, prices, linear_model, polynomial_model, rbf_model):

    plt.scatter(dates, prices, color = 'black', label = 'Data') # plot data points

    # plot our regression models
    plt.plot(dates, linear_model.predict(dates), color = 'red', label = 'Linear Model') 
    plt.plot(dates, polynomial_model.predict(dates), color = 'green', label = 'Polynomial Model')
    plt.plot(dates, rbf_model.predict(dates), color = 'blue', label = 'RBF Model')

    # labes our axes
    plt.xlabel('Date')
    plt.ylabel('Price')

    # finalize our graph and produce plot
    plt.title('Stock Price Prediction (SVR)')
    plt.legend()
    plt.show()

def process_data(filename, x):
    
    # get our dates and prices from csv file
    dates, prices = get_data(filename)

    # train our models
    linear_model, polynomial_model, rbf_model = train_models(dates, prices)

    #plot our results
    plot_results(dates, prices, linear_model, polynomial_model, rbf_model)

    # prepare our predictions
    x_reshape = np.reshape(x, (1, 1))
    linear_prediction = linear_model.predict(x_reshape)[0]
    polynomial_prediction = polynomial_model.predict(x_reshape)[0]
    rbf_prediction = rbf_model.predict(x_reshape)[0]

    return linear_prediction, polynomial_prediction, rbf_prediction
