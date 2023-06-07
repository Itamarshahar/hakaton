import pre_process
if __name__ == '__main__':
    cols_to_remove = []
    pre_process.run_preprocess("./train.feats.csv", "./train.labels.0.csv", cols_to_remove)


