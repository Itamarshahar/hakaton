from pre_process import run_preprocess
if __name__ == '__main__':
    cols_to_remove = []
    run_preprocess("./train.feats.csv", "./train.labels.0.csv", cols_to_remove)
    cols_to_remove = []
    link1 = "./train.feats.csv"
    link2 = "./train.labels.0.csv"
    run_preprocess(link1, link2, cols_to_remove)
    #

