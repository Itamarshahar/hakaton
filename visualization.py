from pre_process import prepreprocess, run_preprocess
import numpy as np
import plotly.express as px
import pandas as pd
from typing import NoReturn, Optional
import matplotlib.pyplot as plt
from predicting_metastases import PredictingMetastases
def draw(X, y) -> None:
    """

    """
    values = []
    for val in values:
        plot_corelation(X,y, val)
def generate_is_sick_vector(y):
    is_sick_vector = np.where(y.sum(axis=1) > 0, 1, 0)
    is_sick_vector = pd.DataFrame(is_sick_vector, columns=['sick'])
    return is_sick_vector
def plot_corelation(X, y, val):
    val_counts = {}
    is_sick = generate_is_sick_vector(y)
    X["is_sick"] = is_sick # Filter X based on is_sick
    for val in X['Nodesexam'].unique():
        count = ((X['Nodesexam'] == val) & (X['is_sick'] == 1)).sum()
        val_counts[val] = count
    x_vals = list(val_counts.values())
    y_vals = list(val_counts.keys())

    # Plot the values
    plt.bar(x_vals, y_vals)
    plt.ylabel("Amount of sick persons")
    plt.xlabel("Nodes exam value")
    plt.title("Correlation Plot")
    plt.show()


def catagorial_label_perc(data: pd.DataFrame, response: pd.DataFrame,  cancer_site: str, orig_col : str):
    probabilities = []

    column_names = data.columns
    stage_columns = [col for col in column_names if orig_col in col.lower()]

    data = data.append(response)

    for column in column_names:
        filtered_data = data[data[column] == 1]  # Filter dataframe to include only rows where the value is 1
        probability = filtered_data[filtered_data[cancer_site] == 1].count() / filtered_data.shape[0] # Calculate the probability
        probabilities.append(probability)
    fig = px.bar(x=stage_columns, y=probabilities, labels={'x': 'Columns', 'y': 'Probability of 1'})
    fig.update_layout(title='Probability of Having a Value of 1 in Each Column')
    fig.show()

def model_selection(X,y):
    k_range = list(range(1, 40, 2))
    n = int(X.shape[0]*0.5)
    X_train_smaller, y_train_smaller = X[:n], X[:n]
    X_val, y_val = X[n:], X[n:]


    # Train and evaluate models for all values of k
    train_errors, val_errors, test_errors = [], [], []
    for k in k_range:
        model = PredictingMetastases(k).fit(X_train_smaller, y_train_smaller)
        train_errors.append(1 - model.score(X_train_smaller, y_train_smaller))
        val_errors.append(1 - model.score(X_val, y_val))
        test_errors.append(1-model.score(X_test, y_test))


    # Select model with lowest training error
    min_ind = np.argmin(np.array(val_errors))
    selected_k = np.array(k_range)[min_ind]
    selected_error = val_errors[min_ind]


    # Plot train- and test errors as well as which model (value of k) was selected
    fig = go.Figure([
        go.Scatter(name='Train Error', x=k_range, y=train_errors, mode='markers+lines', marker_color='rgb(152,171,150)'),
        go.Scatter(name='Validation Error', x=k_range, y=val_errors, mode='markers+lines', marker_color='rgb(220,179,144)'),
        go.Scatter(name='Test Error', x=k_range, y=test_errors, mode='markers+lines', marker_color='rgb(25,115,132)'),
        go.Scatter(name='Selected Model', x=[selected_k], y=[selected_error], mode='markers', marker=dict(color='darkred', symbol="x", size=10))
    ]).update_layout(title=r"$\text{(2) }k\text{-NN Errors - Selection By Minimal Error Over Validation Set}$",
                     xaxis_title=r"$k\text{ - Number of Neighbors}$",
                     yaxis_title=r"$\text{Error Value}$").show()