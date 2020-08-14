# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:49:27 2020

@author: edvonaldo
"""


# Pandas is used for data manipulation
import pandas as pd

# Read in data as pandas dataframe and display first 5 rows
features = pd.read_csv('C:/Users/edvon/Desktop/temps.csv')
#features.head(5)

import datetime

# Get years, months, and days
years = features['year']
months = features['month']
days = features['day']

# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# One-hot encode categorical features
features = pd.get_dummies(features)
features.head(5)

# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
labels = np.array(features['actual'])

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('actual', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25,
                                                                           random_state = 42)

# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('average')]

# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2), 'degrees.')

#%% DT

# Import the model we are using
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

n_cv = int(10)

# Instantiate model 
dt = DecisionTreeRegressor(random_state=42)

# Train the model on training data
dt.fit(train_features, train_labels);

accuracy_cv = cross_val_score(dt, train_features, train_labels, cv=n_cv)

# Use the forest's predict method on the test data
predictions = dt.predict(test_features)

# Calculate the absolute errors
errors_mae = mean_absolute_error(test_labels, predictions)
errors_mse = mean_squared_error(test_labels, predictions)
arrors_r2 = r2_score(test_labels, predictions)

accuracy_dt = dt.score(test_features, test_labels)

# Print out the mean absolute error (mae)
print('MAE:', round(np.mean(errors_mae), 2), 'degrees.')
print('MSE:', round(np.mean(errors_mse), 2), 'degrees.')
print('Accuracy DT:', round(np.mean(accuracy_dt), 2))
print('Accuracy CV:', round(np.mean(accuracy_cv), 2))
print('Accuracy R2:', round(np.mean(accuracy_dt), 2))