import numpy as np
import sklearn.linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

from pre_process import run_preprocess, make_unique_response
from visualization import draw
from predicting_tumor_size import PredictTumorSize
SAMPLE_PATH_10 = "./Data/DATA_by_percent_THIS_IS_GOOD/10_percent_train/10_train.feats.csv"
LABEL_PATH_10 = "./Data/DATA_by_percent_THIS_IS_GOOD/10_percent_train/10_train.labels.0.csv"


SAMPLE_PATH_20 = "./Data/DATA_by_percent_THIS_IS_GOOD/20_percent_train/20_train.feats.csv"
LABEL_PATH_20 = "./Data/DATA_by_percent_THIS_IS_GOOD/20_percent_train/20_train.labels.0.csv"
LABEL_PATH_20_TUMOR_SIZE = "./Data/DATA_by_percent_THIS_IS_GOOD/20_percent_train/20_train.labels.1.csv"

LABEL_PATH_40_TUMOR_SIZE = "./Data/DATA_by_percent_THIS_IS_GOOD/40_percent_train/40_train.labels.1.csv"
LABEL_PATH_40 = "./Data/DATA_by_percent_THIS_IS_GOOD/40_percent_train/40_train.labels.0.csv"
SAMPLE_PATH_40 = "./Data/DATA_by_percent_THIS_IS_GOOD/40_percent_train/40_train.feats.csv"

def testing_tumor_size(str1, str2, lst1, lst2):
    hillel_X, hillel_y = run_preprocess(str1, str2, lst1, lst2, mode="tumor_size")
    # h_train_x, h_test_x, h_train_y, h_test_y = train_test_split(hillel_X, hillel_y, test_size=0.2)
    h_train_x, h_test_x, h_train_y, h_test_y = train_test_split(hillel_X, hillel_y, test_size=0.2)
    h_train_x = h_train_x.to_numpy()
    h_test_x = h_test_x.to_numpy()
    h_train_y = h_train_y.to_numpy()
    h_test_y = h_test_y.to_numpy()
    data = np.column_stack((h_train_x, h_train_y))

    # Compute the correlation matrix
    corr_matrix = np.corrcoef(data, rowvar=False)


    # Plot the correlation matrix using a heatmap
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=np.arange(h_train_x.shape[1]), yticklabels=["h_train_y"] + list(np.arange(h_train_x.shape[1]).astype(str)))
    plt.show()
    # np.random.shuffle(h_train_y)
    # print(sklearn.linear_model.LinearRegression().fit(h_train_x, h_train_y).score(h_test_x,h_test_y))
    # print(sklearn.linear_model.Ridge().fit(h_train_x, h_train_y).score(h_test_x,h_test_y))
    for i in range(h_train_x.shape[1]):
        for j in range(i):
            X = h_train_x[:, j:i]
            learner = PredictTumorSize()
            learner._fit(X, h_train_y)
            res = learner._loss(h_test_x[:, j:i], h_test_y)
            print("i,j =" ,i, ",", j, "res = ", res)
        pass
    learner = PredictTumorSize()
    learner._fit(h_train_x, h_train_y)
    res = learner._loss(h_test_x, h_test_y)
    return res

if __name__ == '__main__':
    np.random.seed(0)
    cols_to_remove = []
    #run_preprocess("./train.feats.csv", "./train.labels.0.csv", cols_to_remove)
    cols_to_remove = []
    run_preprocess(SAMPLE_PATH_20, LABEL_PATH_20,['Age'], ["Surgeryname1",'FormName','Basicstage', 'Hospital'])
    X, y = run_preprocess(SAMPLE_PATH_20, LABEL_PATH_20,['Age'], ["Surgeryname1",'FormName','Basicstage'])

    print(testing_tumor_size(SAMPLE_PATH_40, LABEL_PATH_40_TUMOR_SIZE,['Age'], ["Surgeryname1",'FormName','Basicstage']))
    draw(X, make_unique_response(y))

