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
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.8, random_state=42)
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
        convert_to_dummies(X_train, col)
        X_train = X_train.drop(col, axis=1)
    return X_train


def convert_to_dummies(X_train, col_to_dummies, splitter:str = "+"):
    X_train[col_to_dummies] = X_train[col_to_dummies].str.replace(' ', '_')
    unique_words = set()
    for text in X_train[col_to_dummies]:
        words = text.lower().split(splitter)
        unique_words.update(words)
    df = pd.DataFrame()
    for word in unique_words:
        X_train[word] = [int(word in text.lower().split()) for text in X_train[col_to_dummies]]
    return X_train

def make_unique_response(responses: pd.DataFrame) -> pd.DataFrame:
    col_name = responses.columns[0]
    responses = responses.applymap(clean_responses)
    return convert_to_dummies(responses, col_name, splitter=",")


def clean_responses(reponse:str):
    matches = re.findall(r"'(.*?)'", reponse)
    return ','.join(matches)
def run_preprocess(samples_file_name: str, responses_file_name: str, cols_to_remove:[str], cols_to_dummies:[str]):
    X_train, X_test, y_train, y_test = load_data(samples_file_name, responses_file_name)
    X_train = prepreprocess(X_train, y_train, cols_to_remove, cols_to_dummies)
    make_unique_response(y_train)
    non_numeric_cols = X_train.select_dtypes(exclude=[np.number]).columns
    X_train_numeric_only = X_train.drop(non_numeric_cols, axis=1)
    X_train_numeric_only.fillna(0, inplace=True)
    return X_train_numeric_only

