from sklearn.model_selection import train_test_split
from pca_visualization import run_pca_visualisation
from pre_process import run_preprocess, make_unique_response
from predicting_metastases import PredictingMetastases
from visualization import draw, catagorial_label_perc, generate_is_sick_vector
import predicting_tumor_size
SAMPLE_PATH_10 = "./Data/DATA_by_percent_THIS_IS_GOOD/10_percent_train/10_train.feats.csv"
LABEL_PATH_10 = "./Data/DATA_by_percent_THIS_IS_GOOD/10_percent_train/10_train.labels.0.csv"

SAMPLE_PATH_20 = "./Data/DATA_by_percent_THIS_IS_GOOD/20_percent_train/20_train.feats.csv"
LABEL_PATH_20 = "./Data/DATA_by_percent_THIS_IS_GOOD/20_percent_train/20_train.labels.0.csv"

COLS_TO_DUM = ['FormName','Basicstage', 'Hospital',
               'UserName','Histologicaldiagnosis', 'Histopatologicaldegree','N-lymphnodesmark(TNM)','Stage',
            'Surgeryname1', 'Surgeryname2', 'Surgeryname3', 'T-Tumormark(TNM)', 'surgerybeforeorafter-Actualactivity']

COL_TO_REMOVE = ['Diagnosisdate', 'Surgerydate1', 'Surgerydate2','Surgerydate3','surgerybeforeorafter-Activitydate']

def run_tumor_size(str1, str2, lst1, lst2):
    hillel_X, hillel_y = run_preprocess(str1, str2, lst1, lst2, mode="tumor_size")
    h_train_x, h_train_y, h_test_x, h_test_y = train_test_split(hillel_X, hillel_y, test_size=0.2)
    learner = predicting_tumor_size.PredictTumorSize()
    learner._fit(h_train_x, h_train_y)
    res = learner._loss(h_test_x, h_train_y)
    return res


def run_metastases(X,y):
    X = X.values
    y = y.values
    # train_x, test_x,train_y, test_y = train_test_split(X, y, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    metastases_model = PredictingMetastases()
    metastases_model._fit(X_train, y_train)
    loss = metastases_model._loss(X_test, y_test)
    print(loss)

if __name__ == '__main__':
    cols_to_remove = []
    #run_preprocess("./train.feats.csv", "./train.labels.0.csv", cols_to_remove)
    cols_to_remove = []
    X, y = run_preprocess(SAMPLE_PATH_20, LABEL_PATH_20,COL_TO_REMOVE, COLS_TO_DUM)
    #print(testing_tumor_size(SAMPLE_PATH_20, LABEL_PATH_20,COL_TO_REMOVE, COLS_TO_DUM))
    # for col in COLS_TO_DUM:
    #     catagorial_label_perc(X, generate_is_sick_vector(y), col)

    run_metastases(X,y)
    # y = y.values
    #
    # run_pca_visualisation(X,y)
#

