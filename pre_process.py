import re
from typing import NoReturn
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

convert_to_mean = ['Age']


def load_data(samples_file_name: str, responses_file_name: str) :
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # load data
    raw_data_x = pd.read_csv(samples_file_name)
    raw_data_y = pd.read_csv(responses_file_name)
    X_train, X_test, y_train, y_test = train_test_split(raw_data_x, raw_data_y,test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def prepreprocess(X_train: pd.DataFrame, y_train: pd.DataFrame, cols_to_remove: [str], cols_to_dummies: [str]):

    # Initialize CountVectorizer
    vectorizer = CountVectorizer()

    # Fit and transform the text data
    X_train.rename(columns=lambda x: x.replace(' ', ''), inplace=True)
    X_train.rename(columns=lambda x: x.replace("אבחנה-", ''), inplace=True)
    y_train.rename(columns=lambda x: x.replace("אבחנה-", ''), inplace=True)
    names = X_train.columns
    #need to convert nans
    X_train['Surgeryname1'].fillna("NA", inplace= True)

    #removes all unwanted cols
    for col in cols_to_remove:
        X_train = X_train.drop(col, axis= 1)

    for col in cols_to_dummies:
        X_train = convert_to_dummies(X_train, col)

    return X_train

def change_value(df : pd.DataFrame, col_name:str , convert_dict: dict[str,int], default_value: any= 0):
    look_for_key = convert_dict.keys()
    look_for_value = convert_dict.values()
    col = df.applymap(lambda x:x.lower() if isinstance(x, str) else x)[col_name]
    #col.fillna(default_value)
    #col = col.replace({val: convert_dict.get(val, default_value) for val in col.unique()})
    for key in look_for_key:
        col = col.replace({r'{}'.format(key)}, convert_dict[key] ,regex = True)
    col = np.where(~np.isin(col, look_for_value), col, default_value)
    df[col_name] = col
    return df
def convert_to_dummies(df, col_to_dummies, splitter:str = "+"):
    df[col_to_dummies] = df[col_to_dummies].astype(str)
    df[col_to_dummies] = df[col_to_dummies].str.replace(' ', '_')
    unique_words = set()
    for text in df[col_to_dummies]:
        words = text.lower().split(splitter)
        if words[0] != '':
            unique_words.update(words)
    for word in unique_words:
        df[col_to_dummies + " " + word] = [int(word in text.lower().split()) for text in df[col_to_dummies]]
    return df.drop(col_to_dummies, axis= 1)

def make_unique_response(responses: pd.DataFrame) -> pd.DataFrame:
    col_name = responses.columns[0]
    responses = responses.applymap(clean_responses)
    return convert_to_dummies(responses, col_name, splitter=",")

def clean_responses(response:str):
    response = str(response)
    matches = re.findall(r"'(.*?)'", response)
    return ','.join(matches)
def run_preprocess(samples_file_name: str, responses_file_name: str, cols_to_remove:[str], cols_to_dummies:[str], mode: str='meta'):
    """
    return matrix of only numbers
    """
    X_train, X_test, y_train, y_test = load_data(samples_file_name, responses_file_name)
    X_train = prepreprocess(X_train, y_train, cols_to_remove, cols_to_dummies)
    if mode == 'meta':
        y_train = make_unique_response(y_train)
    for col in convert_to_mean:
        X_train[col].fillna(X_train[col].mean())
    X_train = change_value(X_train,'Histopatologicaldegree',{"null":-1, "gx":0, "g1":1,"g2":2, "g3":3, "g4":4 }, default_value= 0)
    X_train = treat_IVI(X_train)
    X_train = trea_M_meta(X_train)
    X_train = treat_Margin_Type(X_train)
    treat_Node_Exam(X_train)
    treat_pos_nodes(X_train)
    X_train['Side'] = np.where(X_train['Side'] == "דו צדדי", "ימין+שמאל", X_train['Side'])
    # X_train.fillna()
    X_train = convert_to_dummies(X_train,'Side')
    X_train = treat_stage(X_train)

    X_train = change_value(X_train, 'pr', {"חיובי": 1, "שלילי": -1, "pos": 1, "neg": -1, "%":1},
                           default_value=0)
    X_train['pr'].fillna(0, inplace=True)
    X_train['pr'] = pd.to_numeric(X_train['pr'],
                                             errors='coerce').fillna(0).astype(float)
    X_train = change_value(X_train, 'pr', {"חיובי": 1, "שלילי": -1, "pos": 1, "neg": -1, "%": 1},
                           default_value=0)
    X_train['er'].fillna(0, inplace=True)
    X_train['er'] = pd.to_numeric(X_train['er'],
                                  errors='coerce').fillna(0).astype(float)

    non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns
    X_train_numeric_only = X_train.drop(non_numeric_cols, axis=1)
    X_train_numeric_only.fillna(0, inplace=True)

    return X_train_numeric_only, y_train


def treat_stage(X_train):
    X_train = change_value(X_train, 'Stage',
                           {"stage0a": 1, "stage0is": 2, "stage0b": 2, "stage1b": 4, "stage1c": 5, "stage1": 3,
                            "stage2b": 7, "stage2c": 8, "stage2": 6, "stage3b": 10, "stage3c": 11, "stage3": 9,
                            "stage4": 12})
    X_train['Stage'].fillna(0, inplace=True)
    X_train['Stage'] = pd.to_numeric(X_train['Stage'],
                                     errors='coerce').fillna(0).astype(float)
    return X_train


def treat_pos_nodes(X_train):
    X_train['Positivenodes'].fillna(0, inplace=True)
    X_train['Positivenodes'] = pd.to_numeric(X_train['Positivenodes'],
                                             errors='coerce').fillna(0).astype(float)

def treat_Node_Exam(X_train):
    X_train['Nodesexam'].fillna(0, inplace=True)
    X_train['Nodesexam'] = pd.to_numeric(X_train['Nodesexam'],
                                         errors='coerce').fillna(0).astype(float)


def treat_Margin_Type(X_train):
    X_train = change_value(X_train, 'MarginType', {"נקיים": -1, "נגועים": 1, "ללא": 0},
                           default_value=0)
    X_train['MarginType'] = pd.to_numeric(X_train['MarginType'],
                                          errors='coerce').fillna(0).astype(int)
    return X_train


def trea_M_meta(X_train):
    X_train = change_value(X_train, 'M-metastasesmark(TNM)', {"m0": 1, "m1a": 3, "m1b": 4, "m1": 2, "mx": 5},
                           default_value=0)
    X_train['M-metastasesmark(TNM)'] = pd.to_numeric(X_train['M-metastasesmark(TNM)'],
                                                     errors='coerce').fillna(0).astype(int)
    return X_train


def treat_IVI(X_train):
    X_train = change_value(X_train, 'Ivi-Lymphovascularinvasion',
                           {"nan": 0, "no": -1, "(-)": -1, "neg": -1, "none": -1, "not": -1, "yes": 1}, default_value=0)
    X_train['Ivi-Lymphovascularinvasion'].replace("+", 0, inplace=True)
    X_train['Ivi-Lymphovascularinvasion'].replace("(+)", 0, inplace=True)
    X_train['Ivi-Lymphovascularinvasion'] = pd.to_numeric(X_train['Ivi-Lymphovascularinvasion'],
                                                          errors='coerce').fillna(0).astype(int)
    return X_train

