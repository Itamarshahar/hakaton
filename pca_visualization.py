# Authors: Vlad Niculae, Mathieu Blondel
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from pre_process import run_preprocess, make_unique_response, run_preocces_only_X
from visualization import draw, catagorial_label_perc, generate_is_sick_vector
import predicting_tumor_size
from predicting_metastases import PredictingMetastases
import numpy as np
import post_process
import scipy.stats as stats

def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.plot(xx, yy, linestyle, label=label)


def plot_subfigure(X, Y, subplot, title, transform):
    if transform == "pca":
        X = PCA(n_components=3).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=3).fit(X, Y).transform(X)
    else:
        raise ValueError

    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    classif = OneVsRestClassifier(SVC(kernel="linear"))
    classif.fit(X, Y)

    plt.subplot(2, 2, subplot)
    plt.title(title)
    colors =['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in range(3):
        zero_class = np.where(Y[:, i])
        # one_class = np.where(Y[:, 1])
        plt.scatter(X[:, 0], X[:, 1], s=40, c="gray", edgecolors=(0, 0, 0))
        plt.scatter(
            X[zero_class, 0],
            X[zero_class, 2],
            s=i*10,
            edgecolors=colors[i],
            facecolors="none",
            linewidths=2,
            label=f"Class {i}",
        )
        # plt.scatter(
        #     X[one_class, 0],
        #     X[one_class, 1],
        #     s=80,
        #     edgecolors="orange",
        #     facecolors="none",
        #     linewidths=2,
        #     label="Class 2",
        # )
        plot_hyperplane(
            classif.estimators_[i], min_x, max_x, "k--", f"Boundary\nfor class {i}"
        )
        # plot_hyperplane(
        #     classif.estimators_[1], min_x, max_x, "k-.", "Boundary\nfor class 2"
        # )
        plt.xticks(())
        plt.yticks(())

        plt.xlim(min_x - 0.5 * max_x, max_x + 0.5 * max_x)
        plt.ylim(min_y - 0.5 * max_y, max_y + 0.5 * max_y)
    if subplot == 2:
        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")
        plt.legend(loc="upper left")



def run_pca_visualisation(X,Y):
    plt.figure(figsize=(8, 6))

    # X, Y = make_multilabel_classification(
    #     n_classes=6, n_labels=1, allow_unlabeled=True, random_state=1
    # )

    # Y = Y[:, ~np.all(Y == 0, axis=0)]
    plot_subfigure(X, Y, 1, "With unlabeled samples + CCA", "cca")
    plot_subfigure(X, Y, 2, "With unlabeled samples + PCA", "pca")

    # X, Y = make_multilabel_classification(
    #     n_classes=2, n_labels=1, allow_unlabeled=False, random_state=1
    # )

    plot_subfigure(X, Y, 3, "Without unlabeled samples + CCA", "cca")
    plot_subfigure(X, Y, 4, "Without unlabeled samples + PCA", "pca")

    plt.subplots_adjust(0.04, 0.02, 0.97, 0.94, 0.09, 0.2)
    plt.show()



def run_metastases(X,y):
    X = X.values
    y = y.values
    # train_x, test_x,train_y, test_y = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
    # run_model_selection(X_train, X_test, y_train, y_test)

    metastases_model = PredictingMetastases()
    metastases_model._fit(X_train, y_train)
    loss = metastases_model._loss(X_test, y_test)
    print(loss)
    return y


if __name__ == '__main__':
    np.random.seed(0)

    COLS_TO_DUM = ['FormName', 'Basicstage', 'Hospital'
        , 'Histologicaldiagnosis', 'N-lymphnodesmark(TNM)',
                   'surgerybeforeorafter-Actualactivity']

    COL_TO_REMOVE = ['UserName', 'Diagnosisdate', 'Surgerydate1',
                     'Surgerydate2', 'Surgerydate3',
                     'surgerybeforeorafter-Activitydate',
                     'KI67protein', 'Surgeryname1', 'Surgeryname2',
                     'Surgeryname3']
    link1 = "/Users/itamar_shahar/PycharmProjects/hakaton/Data/original_data_DONT_TUOCH!!!/train.feats.csv"
    link2 = "/Users/itamar_shahar/PycharmProjects/hakaton/Data/original_data_DONT_TUOCH!!!/train.labels.0.csv"
    X, y = run_preprocess(link1,link2, COL_TO_REMOVE, COLS_TO_DUM, "meta")

    run_pca_visualisation(X,y)