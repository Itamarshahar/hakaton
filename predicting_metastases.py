import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


class PredictingMetastases(BaseEstimator):
    def __int__(self):
        self.logistic_learners_ = []
        self.random_forest = RandomForestClassifier(class_weight="balanced")
        self.fitted = False

    def _fit(self, X, y):
        self.random_forest.fit(X, y)

    def _predict(self, X):
        if not self.fitted:
            print("Must Fit First!")
            return
        res = self.random_forest.predict(X)

    def _loss(self, X, true_y):
        prediction = self._predict(X)
        # prediction[prediction > 1] = 10
        res = sklearn.metrics.accuracy_score(true_y, prediction)
        # print("res is: " + str(res))
        # print("zeros MSE is: " + str(zeros))
        return 1 - res
