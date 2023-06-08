from sklearn.model_selection import train_test_split

from pre_process import run_preprocess, make_unique_response
from visualization import draw
import predicting_tumor_size
SAMPLE_PATH_10 = "./Data/DATA_by_percent_THIS_IS_GOOD/10_percent_train/10_train.feats.csv"
LABEL_PATH_10 = "./Data/DATA_by_percent_THIS_IS_GOOD/10_percent_train/10_train.labels.0.csv"

SAMPLE_PATH_20 = "./Data/DATA_by_percent_THIS_IS_GOOD/20_percent_train/20_train.feats.csv"
LABEL_PATH_20 = "./Data/DATA_by_percent_THIS_IS_GOOD/20_percent_train/20_train.labels.0.csv"

def testing_tumor_size(str1, str2, lst1, lst2):
    hillel_X, hillel_y = run_preprocess(str1, str2, lst1, lst2, mode="tumor_size")
    h_train_x, h_train_y, h_test_x, h_test_y = train_test_split(hillel_X, hillel_y, test_size=0.2)
    learner = predicting_tumor_size.PredictTumorSize()
    learner._fit(h_train_x, h_train_y)
    res = learner._loss(h_test_x, h_train_y)
    return res

if __name__ == '__main__':
    cols_to_remove = []
    #run_preprocess("./train.feats.csv", "./train.labels.0.csv", cols_to_remove)
    cols_to_remove = []
    run_preprocess(SAMPLE_PATH_20, LABEL_PATH_20,['Age'], ["Surgeryname1",'FormName','Basicstage', 'Hospital'])
    X, y = run_preprocess(SAMPLE_PATH_20, LABEL_PATH_20,['Age'], ["Surgeryname1",'FormName','Basicstage'])
    print(testing_tumor_size(SAMPLE_PATH_20, LABEL_PATH_20,['Age'], ["Surgeryname1",'FormName','Basicstage']))
    draw(X, make_unique_response(y))

