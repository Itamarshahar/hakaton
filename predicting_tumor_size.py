import sklearn.linear_model
from sklearn.base import BaseEstimator
from sklearn import linear_model

base_line_ridge_lambda = 0.001  ## todo: GLOBAL!!!!


class PredictTumorSize(BaseEstimator):
    def __init__(self):
        super().__init__()
        self.base_line_ridge = linear_model.Ridge(alpha=base_line_ridge_lambda)
        self.base_line_linear = linear_model.LinearRegression()
        self.is_fit = False

    def _fit(self, X, y):
        self.base_line_ridge.fit(X, y)
        self.base_line_linear.fit(X, y)
        self.is_fit = True

    def _predict(self, X):
        if not self.is_fit:
            print("MUST FIT FIRST!")
        ridge = self.base_line_ridge.predict(X)
        linear = self.base_line_linear.predict(X)
        return ridge

    def _loss(self, X, true_y):
        prediction = self._predict(X)
        res = sklearn.metrics.mean_squared_error(true_y, prediction)
        print("res is: " + str(res))
        return res
