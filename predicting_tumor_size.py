import sklearn.linear_model
from sklearn.base import BaseEstimator
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
base_line_ridge_lambda = 0.5  ## todo: GLOBAL!!!!
from sklearn.ensemble import RandomForestRegressor

class PredictTumorSize(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=100)
        self.is_fit = False

    def _fit(self, X, y):
        self.model.fit(X, y)
        self.is_fit = True

    def _predict(self, X):
        if not self.is_fit:
            print("MUST FIT FIRST!")
        return self.model.predict(X)

    def _loss(self, X, true_y):
        y_pred = self._predict(X)
        print("mean_sq_err:", mean_squared_error(y_true=true_y, y_pred=y_pred))
        # print(np.sum(y_pred*true_y))
        return mean_squared_error(y_true=true_y, y_pred=y_pred)
