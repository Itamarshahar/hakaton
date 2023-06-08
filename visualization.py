from pre_process import prepreprocess, run_preprocess
import numpy as np
import plotly.express as px
import pandas as pd
from typing import NoReturn, Optional
import matplotlib.pyplot as plt
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

# def feature_evaluation(X: pd.DataFrame, y: pd.Series,
#                     output_path: str = ".") -> NoReturn:
#
#     """
#  162 Create scatter plot between each feature and the response.
#  163 - Plot title specifies feature name
#  164 - Plot title specifies Pearson Correlation between feature and response
#  165 - Plot saved under given folder with file name including feature name
#  166 Parameters
#  167 ----------
#  168 X : DataFrame of shape (n_samples, n_features)
#  169 Design matrix of regression problem
#  170
#  171 y : array-like of shape (n_samples, )
#  172 Response vector to evaluate against
#  173
#  174 output_path: str (default ".")
#  175 Path to folder in which plots are saved
#  176 """
#
#     sigma_y = np.std(y)
#
#     for feature in X:
#         for label in y:
#             # tmp = y[label].unique()
#             sigma_x = np.std(X[feature].astype(float))
#
#             correlation_value = \
#             ((np.cov(X[feature].astype(float), label)) / (sigma_x * sigma_y))[0, 1]
#
#             fig = px.scatter(x=X[feature], y=y[label])
#
#             fig.update_layout(
#                 title = f'The Correlation Between the Feature {feature} Values '
#                 f'with the Responses <br> Pearson Correlation v'
#                     f'alues:{correlation_value}',
#
#                 xaxis_title = f'{feature} values',
#                 yaxis_title = 'Response values (y)')
#
#                 # fig.write_image(output_path + f"/pearson_value_for_{feature}.png")
#             fig.show()#write_image(output_path + f"/pearson_value_for_{feature}.png")


