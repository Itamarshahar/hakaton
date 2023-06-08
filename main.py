import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from pre_process import run_preprocess, make_unique_response
from visualization import draw, catagorial_label_perc, generate_is_sick_vector
import predicting_tumor_size
from predicting_metastases import PredictingMetastases
import numpy as np
import scipy.stats as stats
SAMPLE_PATH_10 = "./Data/DATA_by_percent_THIS_IS_GOOD/10_percent_train/10_train.feats.csv"
LABEL_PATH_10 = "./Data/DATA_by_percent_THIS_IS_GOOD/10_percent_train/10_train.labels.0.csv"

SAMPLE_PATH_20 = "./Data/DATA_by_percent_THIS_IS_GOOD/20_percent_train/20_train.feats.csv"
LABEL_PATH_20 = "./Data/DATA_by_percent_THIS_IS_GOOD/20_percent_train/20_train.labels.0.csv"


SAMPLE_PATH_60 = "./Data/DATA_by_percent_THIS_IS_GOOD/60_percent_train/60_train.feats.csv"
LABEL_PATH_60 = "./Data/DATA_by_percent_THIS_IS_GOOD/60_percent_train/60_train.labels.0.csv"
LABEL1_PATH_60 = "./Data/DATA_by_percent_THIS_IS_GOOD/60_percent_train/60_train.labels.1.csv"
COLS_TO_DUM = ['FormName','Basicstage', 'Hospital',
               'UserName','Histologicaldiagnosis','N-lymphnodesmark(TNM)',
             'surgerybeforeorafter-Actualactivity']

COL_TO_REMOVE = ['Diagnosisdate', 'Surgerydate1', 'Surgerydate2','Surgerydate3','surgerybeforeorafter-Activitydate',
                 'KI67protein','Surgeryname1', 'Surgeryname2', 'Surgeryname3']

def run_tumor_size(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    learner = predicting_tumor_size.PredictTumorSize()
    learner._fit(X_train, y_train)
    return learner._loss(X_test, y_test)

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

def get_column_names_with_ones(y: np.ndarray, col_names: [str]):
    result = []
    for row in range(y.shape[0]):
        row_res = []
        for col in range(y.shape[1]):
            if y[row][col] == 1:
                row_res.append(col_names[col])
        result.append(row_res)
    return pd.DataFrame(result)




if __name__ == '__main__':
    np.random.seed(0)

    cols_to_remove = []
    #run_preprocess("./train.feats.csv", "./train.labels.0.csv", cols_to_remove)

    cols_to_remove = []

    # X, y = run_preprocess("/Users/itamar_shahar/PycharmProjects/hakaton/Data/original_data_DONT_TUOCH!!!/train.feats.csv", "/Users/itamar_shahar/PycharmProjects/hakaton/Data/original_data_DONT_TUOCH!!!/train.labels.0.csv",COL_TO_REMOVE, COLS_TO_DUM)
    # X, y1 = run_preprocess(SAMPLE_PATH_60, LABEL_PATH_60,COL_TO_REMOVE, COLS_TO_DUM)
    # X, y2 = run_preprocess(SAMPLE_PATH_60, LABEL_PATH_60,COL_TO_REMOVE, COLS_TO_DUM,"meta")
    #
    # run_metastases(X,y)
    # respon=get_column_names_with_ones(run_metastases(X,y2), y2.columns)
    # run_tumor_size(SAMPLE_PATH_60, LABEL_PATH_60,COL_TO_REMOVE, COLS_TO_DUM)
    # respon.to_excel('./output.xlsx', index=False)

    X, y = run_preprocess(SAMPLE_PATH_60, LABEL1_PATH_60,COL_TO_REMOVE, COLS_TO_DUM)
    X.to_csv("/Users/itamar_shahar/PycharmProjects/hakaton/X.csv")
    y.to_csv("/Users/itamar_shahar/PycharmProjects/hakaton/y.csv")
    X = pd.read_csv("/Users/itamar_shahar/PycharmProjects/hakaton/X.csv")
    y = pd.read_csv("/Users/itamar_shahar/PycharmProjects/hakaton/y.csv")

    run_tumor_size(X, y)
    # for col in COLS_TO_DUM:
    #     catagorial_label_perc(X, generate_is_sick_vector(y), col)
    # draw(X, make_unique_response(y))

