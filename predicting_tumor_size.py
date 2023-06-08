import sklearn.linear_model
from sklearn.base import BaseEstimator
from sklearn import linear_model
import numpy as np

base_line_ridge_lambda = 0.5  ## todo: GLOBAL!!!!


class PredictTumorSize(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.base_line_ridge = linear_model.Ridge(alpha=base_line_ridge_lambda)
        self.base_line_linear = linear_model.LinearRegression()
        self.base_logistic = linear_model.LogisticRegression()
        self.is_fit = False

    def _fit(self, X, y):
        self.base_line_ridge.fit(X, y)
        self.base_line_linear.fit(X, y)
        # self.base_logistic.fit(X, y)
        self.is_fit = True

    def _predict(self, X):
        if not self.is_fit:
            print("MUST FIT FIRST!")
        # ridge = self.base_line_ridge.predict(X)
        linear = self.base_line_linear.predict(X)
        # logistic = self.base_logistic.predict(X)
        return linear

    def _loss(self, X, true_y):
        prediction = self._predict(X)
        # prediction[prediction > 1] = 10
        res = sklearn.metrics.mean_squared_error(true_y, prediction)
        # print("res is: " + str(res))
        zeros = sklearn.metrics.mean_squared_error(true_y, np.ones(true_y.shape[0]))
        # print("zeros MSE is: " + str(zeros))
        return res
