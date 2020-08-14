# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:56:30 2020

@author: edvonaldo
"""

# Pandas is used for data manipulation
import pandas as pd

# Read in data as pandas dataframe and display first 5 rows
features = pd.read_csv('C:/Users/edvon/Desktop/random_forest_explained/data/temps_set.csv')
features.head(5)

#%%
print('The shape of our features is:', features.shape)
# Descriptive statistics for each column
features.describe()

#%%
import datetime

# Get years, months, and days
years = features['year']
months = features['month']
days = features['day']

# List and then convert to datetime object
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

#%%
# One-hot encode categorical features
features = pd.get_dummies(features)
features.head(5)

#%%
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

#%%
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, \
train_labels, test_labels = train_test_split(features, labels, 
                                             test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

#%%
# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('average')]

# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2), 'degrees.')

#%%
# Import the model we are using
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

n_cv = int(6)

# Instantiate model 
dt = DecisionTreeRegressor()

accur_cv = cross_val_score(dt, train_features, train_labels, cv=n_cv)

# Train the model on training data
dt.fit(train_features, train_labels);

#%%
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Use the forest's predict method on the test data
predictions = dt.predict(test_features)

# Calculate and display accuracy
accuracy = r2_score(test_labels, predictions)
accuracy_dt = dt.score(test_features, test_labels)
accuracy_mse = mean_squared_error(test_labels, predictions)
accuracy_mae = mean_absolute_error(test_labels, predictions)

print('No DT: ', round(accuracy, 2))
print('Yes DT:', round(accuracy_dt, 2))
print('MSE: ', round(accuracy_mse, 2))
print('MAE:', round(accuracy_mae, 2))
print('CV:', round(accur_cv.mean(), 2))
print("CV array: ", np.round(accur_cv, 2))