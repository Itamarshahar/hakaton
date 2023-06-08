from pre_process import run_preprocess,make_unique_response
from visualization import draw

SAMPLE_PATH = "./Data/DATA_by_percent_THIS_IS_GOOD/10_percent_train/10_train.feats.csv"
LABEL_PATH = "./Data/DATA_by_percent_THIS_IS_GOOD/10_percent_train/10_train.labels.0.csv"

if __name__ == '__main__':
    cols_to_remove = []
    #run_preprocess("./train.feats.csv", "./train.labels.0.csv", cols_to_remove)
    cols_to_remove = []
    X, y = run_preprocess(SAMPLE_PATH, LABEL_PATH,['Age'], ["Surgeryname1",'FormName','Basicstage'])
    draw(X, make_unique_response(y))

