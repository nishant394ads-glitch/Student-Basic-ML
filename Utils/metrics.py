from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

def regression_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def classification_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
