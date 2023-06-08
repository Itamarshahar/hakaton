import numpy as np
import sklearn
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score


class PredictingMetastases(BaseEstimator):
    def __init__(self) -> None :
        self.logistic_learners_ = []

        self.random_forest = RandomForestClassifier(n_estimators=90, random_state=42, class_weight="balanced")
        self.fitted = False
        # self.n_neighbors = KNeighborsClassifier(8)


    def _fit(self, X, y) -> None:
        self.predicted_labels = cross_val_score(self.random_forest, X, y, cv=5)
        self.random_forest.fit(X, y)
        # self.n_neighbors.fit(X, y)
        self.fitted = True

    def _predict(self, X) :
        if not self.fitted:
            print("Must Fit First!")
            return
        return self.random_forest.predict(X)
        # return self.n_neighbors.predict(X)

    def _loss(self, X, true_y):
        y_pred = self._predict(X)
        print("1 in y_pred:", np.sum(y_pred))
        print("1 in y_true:", np.sum(true_y))
        print("2:", np.sum(y_pred * true_y))
        return f1_score(true_y, y_pred, average="micro", zero_division=1), f1_score(true_y, y_pred,  average="macro", zero_division=1)
        # res = sklearn.metrics.accuracy_score(true_y, prediction)
        # print("res is: " + str(res))
        # print("zeros MSE is: " + str(zeros))
        # return 1 - res
