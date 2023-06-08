from sklearn.model_selection import train_test_split
import numpy as np
from pre_process import run_preprocess, make_unique_response
from visualization import draw, catagorial_label_perc, generate_is_sick_vector
import predicting_tumor_size
from predicting_metastases import PredictingMetastases
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import scipy.stats as stats
SAMPLE_PATH_10 = "./Data/DATA_by_percent_THIS_IS_GOOD/10_percent_train/10_train.feats.csv"
LABEL_PATH_10 = "./Data/DATA_by_percent_THIS_IS_GOOD/10_percent_train/10_train.labels.0.csv"

SAMPLE_PATH_20 = "./Data/DATA_by_percent_THIS_IS_GOOD/20_percent_train/20_train.feats.csv"
LABEL_PATH_20 = "./Data/DATA_by_percent_THIS_IS_GOOD/20_percent_train/20_train.labels.0.csv"


SAMPLE_PATH_60 = "./Data/DATA_by_percent_THIS_IS_GOOD/60_percent_train/60_train.feats.csv"
LABEL_PATH_60 = "./Data/DATA_by_percent_THIS_IS_GOOD/60_percent_train/60_train.labels.0.csv"
LABEL1_PATH_60 = "./Data/DATA_by_percent_THIS_IS_GOOD/60_percent_train/60_train.labels.1.csv"

SAMPLE_PATH_80 = "./Data/DATA_by_percent_THIS_IS_GOOD/80_percent_train/80_train.feats.csv"
LABEL_PATH_80 = "./Data/DATA_by_percent_THIS_IS_GOOD/80_percent_train/80_train.labels.0.csv"

COLS_TO_DUM = ['FormName','Basicstage', 'Hospital','Histologicaldiagnosis','N-lymphnodesmark(TNM)',
             'surgerybeforeorafter-Actualactivity']

COL_TO_REMOVE = ['Diagnosisdate', 'Surgerydate1', 'Surgerydate2','Surgerydate3','surgerybeforeorafter-Activitydate',
                 'KI67protein', 'Surgeryname1', 'Surgeryname2', 'Surgeryname3', 'UserName']

def run_tumor_size(str1, str2, lst1, lst2):
    hillel_X, hillel_y = run_preprocess(str1, str2, lst1, lst2, mode="tumor_size")
    h_train_x, h_train_y, h_test_x, h_test_y = train_test_split(hillel_X, hillel_y, test_size=0.2)
    learner = predicting_tumor_size.PredictTumorSize()
    learner._fit(X_train, X_train)
    return learner._loss(X_test, y_test)

def run_metastases(X,y):
    X = X.values
    y = y.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
    # run_model_selection(X_train, X_test, y_train, y_test)

    metastases_model = PredictingMetastases(96)
    metastases_model._fit(X_train, y_train)
    print(metastases_model._loss(X_test, y_test))




    feature_names_sorted = []
    feature_importances_sorted = []
    for feature, importance in sorted_feature_importances:
        feature_names_sorted.append(feature)
        feature_importances_sorted.append(importance)
        print(f"Feature: {feature}, Importance: {importance}")

    fig = px.bar(x=feature_names_sorted, y=feature_importances_sorted, title='Features importance using model weights')
    fig.show()


        # fig = px.bar(sorted_feature_importances, x='Feature', y='Correlation',
        #          title='Feature importance')
    #
    # fig.show()

    return sorted_feature_importances

if __name__ == '__main__':
    np.random.seed(0)

    cols_to_remove = []
    #run_preprocess("./train.feats.csv", "./train.labels.0.csv", cols_to_remove)


    # X, y = run_preprocess("/Users/itamar_shahar/PycharmProjects/hakaton/Data/original_data_DONT_TUOCH!!!/train.feats.csv", "/Users/itamar_shahar/PycharmProjects/hakaton/Data/original_data_DONT_TUOCH!!!/train.labels.0.csv",COL_TO_REMOVE, COLS_TO_DUM)
    # X, y = run_preprocess(SAMPLE_PATH_60, LABEL_PATH_60,COL_TO_REMOVE, COLS_TO_DUM, mode="meta")
    # run_metastases(X,y)
    # generate_best_feature(X, y)

    X, y = run_preprocess(SAMPLE_PATH_60, LABEL1_PATH_60,COL_TO_REMOVE, COLS_TO_DUM)

    run_tumor_size(X,y)
    # for col in COLS_TO_DUM:
    #     catagorial_label_perc(X, generate_is_sick_vector(y), col)
    # draw(X, make_unique_response(y))

