import pre_process
if __name__ == '__main__':
    cols_to_remove = ['FormName', 'Hospital', 'UserName', 'אבחנה-Age', 'אבחנה-Basicstage',
       'אבחנה-Diagnosisdate', 'אבחנה-Her2', 'אבחנה-Histologicaldiagnosis',
       'אבחנה-Histopatologicaldegree', 'אבחנה-Ivi-Lymphovascularinvasion',
       'אבחנה-KI67protein', 'אבחנה-Lymphaticpenetration',
       'אבחנה-M-metastasesmark(TNM)', 'אבחנה-MarginType',
       'אבחנה-N-lymphnodesmark(TNM)', 'אבחנה-Nodesexam', 'אבחנה-Positivenodes',
       'אבחנה-Side', 'אבחנה-Stage', 'אבחנה-Surgerydate1', 'אבחנה-Surgerydate2',
       'אבחנה-Surgerydate3', 'אבחנה-Surgeryname1', 'אבחנה-Surgeryname2',
       'אבחנה-Surgeryname3', 'אבחנה-Surgerysum', 'אבחנה-T-Tumormark(TNM)',
       'אבחנה-Tumordepth', 'אבחנה-Tumorwidth', 'אבחנה-er', 'אבחנה-pr',
       'surgerybeforeorafter-Activitydate',
       'surgerybeforeorafter-Actualactivity', 'id-hushed_internalpatientid']
    pre_process.run_preprocess("./train.feats.csv", "./train.labels.0.csv", cols_to_remove)


