from pre_process import run_preprocess

SAMPLE_PATH_10 = "./Data/DATA_by_percent_THIS_IS_GOOD/10_percent_train/10_train.feats.csv"
LABEL_PATH_10 = "./Data/DATA_by_percent_THIS_IS_GOOD/10_percent_train/10_train.labels.0.csv"

SAMPLE_PATH_20 = "./Data/DATA_by_percent_THIS_IS_GOOD/20_percent_train/20_train.feats.csv"
LABEL_PATH_20 = "./Data/DATA_by_percent_THIS_IS_GOOD/20_percent_train/20_train.labels.0.csv"

if __name__ == '__main__':
    cols_to_remove = []
    #run_preprocess("./train.feats.csv", "./train.labels.0.csv", cols_to_remove)
    cols_to_remove = []
    run_preprocess(SAMPLE_PATH_20, LABEL_PATH_20,['Age'], ["Surgeryname1",'FormName','Basicstage'])


