
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio


def load_data(filename: str, cols_to_drop = None):
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


    # load data and drop unnecessary data
    raw_data = pd.read_csv(filename).drop(cols_to_drop, axis=1)


    #replace zip code col to dummies
    return pd.get_dummies(raw_data, prefix='zipcode_', columns=['zipcode'], dtype=int)
