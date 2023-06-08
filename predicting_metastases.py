import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


class PredictingMetastases(BaseEstimator):
    def __init__(self):
        self.logistic_learners_ = []
        self.random_forest = RandomForestClassifier(class_weight="balanced")
        self.fitted = False

    def _fit(self, X, y):
        self.random_forest.fit(X, y)
        self.fitted = True

    def _predict(self, X):
        if not self.fitted:
            print("Must Fit First!")
            return
        self.random_forest.predict(X)

    def _loss(self, X, true_y):
        y_pred = self._predict(X)
        return f1_score(true_y, y_pred)
        # res = sklearn.metrics.accuracy_score(true_y, prediction)
        # print("res is: " + str(res))
        # print("zeros MSE is: " + str(zeros))
        # return 1 - res
