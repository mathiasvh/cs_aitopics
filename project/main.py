import pandas as pd
pd.set_option('display.max_columns', 500)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics.regression import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

# TODO
# - scaling

data_file = "Affairs.csv"
train_percentage = 0.8

def get_results(ground_truth, prediction):
    r2 = r2_score(ground_truth, prediction)
    mse = mean_squared_error(ground_truth, prediction)
    mae = mean_absolute_error(ground_truth, prediction)
    rmse = sqrt(mean_squared_error(ground_truth, prediction))
    diff_array = np.abs(np.subtract(ground_truth, prediction))
    max_error = np.max(diff_array)
    max_error_index = np.where(diff_array == max_error)[0][0] + 1
    return r2, mse, mae, rmse, max_error, max_error_index

def print_results(ground_truth, prediction):
    r2, mse, mae, rmse, max_error, max_error_index = get_results(ground_truth, prediction)
    allInformation = "r2: " + str(r2)[:6] + " | mse: " + str(mse)[:6] + " | mae: " + str(mae)[:6] + " | rmse: " + str(rmse)[:6] + " | max_error: " + str(max_error)[:6] + " [idx: " + str(max_error_index) + "]"
    print(allInformation)

df = pd.read_csv(data_file, delimiter=',', 
                 usecols=['affairs','gender','age','yearsmarried','children',
                          'religiousness','education','occupation','rating'],
                 converters={'children': lambda s: True if s == "yes" else False})
display(df.head())
print(df.dtypes)

regressor = GradientBoostingRegressor()
# ommit gender and children (as does the reference paper)
X = df[["yearsmarried","age","religiousness","occupation","rating"]].values
y = df['affairs'].values

# shuffle data
order = np.argsort(np.random.random(y.shape))
X = X[order]
y = y[order]

# split train/test
cutoff_idx = int(train_percentage * y.shape[0])
X_train = X[0:cutoff_idx, :]
X_test  = X[cutoff_idx:, :]
y_train = y[0:cutoff_idx]
y_test  = y[cutoff_idx:]


regressor.fit(X_train, y_train)
test_prediction = regressor.predict(X_test)

print_results(y_test, test_prediction)