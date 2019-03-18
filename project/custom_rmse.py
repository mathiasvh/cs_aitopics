from sklearn.metrics.regression import mean_squared_error
import numpy as np

def rmse_cv(y_true, y_pred) :
	assert len(y_true) == len(y_pred)
	y_pred = np.exp(y_pred)
	y_true = np.exp(y_true)
	return np.sqrt(mean_squared_error(y_true,y_pred))