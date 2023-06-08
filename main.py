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

from predicting_tumor_size import PredictTumorSize
SAMPLE_PATH_10 = "./Data/DATA_by_percent_THIS_IS_GOOD/10_percent_train/10_train.feats.csv"
LABEL_PATH_10 = "./Data/DATA_by_percent_THIS_IS_GOOD/10_percent_train/10_train.labels.0.csv"

SAMPLE_PATH_20 = "./Data/DATA_by_percent_THIS_IS_GOOD/20_percent_train/20_train.feats.csv"
LABEL_PATH_20 = "./Data/DATA_by_percent_THIS_IS_GOOD/20_percent_train/20_train.labels.0.csv"


SAMPLE_PATH_60 = "./Data/DATA_by_percent_THIS_IS_GOOD/60_percent_train/60_train.feats.csv"
LABEL_PATH_60 = "./Data/DATA_by_percent_THIS_IS_GOOD/60_percent_train/60_train.labels.0.csv"
LABEL1_PATH_60 = "./Data/DATA_by_percent_THIS_IS_GOOD/60_percent_train/60_train.labels.1.csv"

SAMPLE_PATH_80 = "./Data/DATA_by_percent_THIS_IS_GOOD/80_percent_train/80_train.feats.csv"
LABEL_PATH_80 = "./Data/DATA_by_percent_THIS_IS_GOOD/80_percent_train/80_train.labels.0.csv"


COLS_TO_DUM = ['FormName','Basicstage', 'Hospital'
               ,'Histologicaldiagnosis','N-lymphnodesmark(TNM)',
             'surgerybeforeorafter-Actualactivity']

COL_TO_REMOVE = ['UserName', 'Diagnosisdate', 'Surgerydate1', 'Surgerydate2','Surgerydate3','surgerybeforeorafter-Activitydate',
                 'KI67protein','Surgeryname1', 'Surgeryname2', 'Surgeryname3']

def run_tumor_size(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    learner = PredictTumorSize()
    learner._fit(X_train, y_train)
    predictions = learner._predict(X_test)
    df_predictions = pd.DataFrame(predictions, columns=['אבחנה-Tumor size'])
    df_predictions.to_csv("part2/predictions.csv", index=False)
    # print(tmp)
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


def generate_submition_file():
    submit_tumor()

    # print(tmp)
    # return model_tumor._loss(X_test, y_test)


    # X, y = run_preprocess(link_to_all_data, link_to_all_labels1,COL_TO_REMOVE, COLS_TO_DUM)
def submit_meta():
    link_to_all_data = "/Users/itamar_shahar/PycharmProjects/hakaton/Data/original_data_DONT_TUOCH!!!/train.feats.csv"
    link_to_all_labels1 = "/Users/itamar_shahar/PycharmProjects/hakaton/Data/original_data_DONT_TUOCH!!!/train.labels.0.csv"
    link_to_test_data = "/Users/itamar_shahar/PycharmProjects/hakaton/test.feats.csv"
    ############################################################################
    # X = pd.read_csv(link_to_all_data)
    # y = pd.read_csv(link_to_all_labels1)
    # X_test = pd.read_csv(link_to_test_data)
    ###########################################################################
    # X_train = run_preocces_only_X(X, y, COL_TO_REMOVE,COLS_TO_DUM)
    # X_test = run_preocces_only_X(X_test, y, COL_TO_REMOVE, COLS_TO_DUM)
    # model_tumor = PredictTumorSize()
    # model_tumor._fit(X_train, y)
    # X_test = X_test.reindex(columns=X_train, fill_value=0)
    # predictions = model_tumor._predict(X_test)
    # df_predictions = pd.DataFrame(predictions,
    #                               columns=['אבחנה-Tumor size'])
    # df_predictions.to_csv("part2/predictions.csv", index=False)


def submit_tumor():
    link_to_all_data = "/Users/itamar_shahar/PycharmProjects/hakaton/Data/original_data_DONT_TUOCH!!!/train.feats.csv"
    link_to_all_labels1 = "/Users/itamar_shahar/PycharmProjects/hakaton/Data/original_data_DONT_TUOCH!!!/train.labels.1.csv"
    link_to_test_data = "/Users/itamar_shahar/PycharmProjects/hakaton/test.feats.csv"
    ############################################################################
    X = pd.read_csv(link_to_all_data)
    y = pd.read_csv(link_to_all_labels1)
    X_test = pd.read_csv(link_to_test_data)
    ############################################################################
    X_train = run_preocces_only_X(X, y, COL_TO_REMOVE,COLS_TO_DUM)
    X_test = run_preocces_only_X(X_test, y, COL_TO_REMOVE, COLS_TO_DUM)
    model_tumor = PredictTumorSize()
    model_tumor._fit(X_train, y)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    predictions = model_tumor._predict(X_test)
    df_predictions = pd.DataFrame(predictions,
                                  columns=['אבחנה-Tumor size'])
    df_predictions.to_csv("part2/predictions.csv", index=False)


if __name__ == '__main__':
    generate_submition_file()
    #run_preprocess("./train.feats.csv", "./train.labels.0.csv", cols_to_remove)


    # X, y = run_preprocess("/Users/itamar_shahar/PycharmProjects/hakaton/Data/original_data_DONT_TUOCH!!!/train.feats.csv", "/Users/itamar_shahar/PycharmProjects/hakaton/Data/original_data_DONT_TUOCH!!!/train.labels.0.csv",COL_TO_REMOVE, COLS_TO_DUM)
    # X, y1 = run_preprocess(SAMPLE_PATH_60, LABEL_PATH_60,COL_TO_REMOVE, COLS_TO_DUM)
    # X, y2 = run_preprocess(SAMPLE_PATH_60, LABEL_PATH_60,COL_TO_REMOVE, COLS_TO_DUM,"meta")
    #
    # run_metastases(X,y)
    # respon=get_column_names_with_ones(run_metastases(X,y2), y2.columns)
    # run_tumor_size(SAMPLE_PATH_60, LABEL_PATH_60,COL_TO_REMOVE, COLS_TO_DUM)
    # respon.to_excel('./output.xlsx', index=False)

    # X, y = run_preprocess("/Users/itamar_shahar/PycharmProjects/hakaton/Data/original_data_DONT_TUOCH!!!/train.feats.csv", "/Users/itamar_shahar/PycharmProjects/hakaton/Data/original_data_DONT_TUOCH!!!/train.labels.1.csv",COL_TO_REMOVE, COLS_TO_DUM)
    # X.to_csv("/Users/itamar_shahar/PycharmProjects/hakaton/X.csv")
    # y.to_csv("/Users/itamar_shahar/PycharmProjects/hakaton/y.csv")
    # X = pd.read_csv("/Users/itamar_shahar/PycharmProjects/hakaton/X.csv")
    # y = pd.read_csv("/Users/itamar_shahar/PycharmProjects/hakaton/y.csv")


    # for col in COLS_TO_DUM:
    #     catagorial_label_perc(X, generate_is_sick_vector(y), col)
    # draw(X, make_unique_response(y))

