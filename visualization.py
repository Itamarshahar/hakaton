from pre_process import prepreprocess, run_preprocess
import numpy as np
import plotly.express as px
import pandas as pd
from typing import NoReturn, Optional
def draw(X, y) -> None:
    """

    """
    # print(X)
    # print(y)
    feature_evaluation(X,y)
    for col in X.columns:
        pass
def generate_is_sick_vector(y):
    is_sick_vector = np.where(y.sum(axis=1) > 0, 1, 0)
    is_sick_vector = pd.DataFrame(is_sick_vector, columns=['sick'])
    return is_sick_vector
def plot_corelation(X, y):
    val_counts = {}
    is_sick = generate_is_sick_vector(y)
    X["is_sick"] = is_sick # Filter X based on is_sick
    for val in X['Nodesexam'].unique():
        count = ((X['Nodesexam'] == val) & (is_sick == 1)).sum()
        val_counts[val] = count

    print(val_counts)

def catagorial_label_perc(data: pd.DataFrame, response: pd.DataFrame, orig_col: str, cancer_site: str = "sick", percentage : bool = True):
    res = []

    column_names = data.columns
    stage_columns = [col for col in column_names if orig_col in col and data[data[col] == 1].count().any()]
    tag = "percentage"
    for column in stage_columns:
        filtered_data = pd.concat((pd.DataFrame(data[column]), response), axis=1)
        filtered_data = filtered_data[filtered_data[column] == 1]

        if filtered_data.shape[0] != 0:
            if percentage:

                probability = filtered_data[filtered_data[cancer_site] == 1].shape[0] / filtered_data.shape[0]
                res.append(probability)
            else:
                tag = "sum"
                sum = filtered_data[filtered_data[cancer_site] == 1].shape[0]
                res.append(sum)

    df = pd.DataFrame({'Columns': stage_columns, f'{tag} of {cancer_site}': res})
    fig = px.bar(df, x='Columns', y=f'{tag} of {cancer_site}', labels={'Columns': 'Columns', 'Probability of 1': 'Probability of 1'})
    fig.update_layout(title=f'{tag} of {cancer_site} in different {orig_col}')
    fig.write_image(f"./catagorial_feature_sick_probability/percentage_as_{orig_col}_at_{tag}.png")

def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                    output_path: str = ".") -> NoReturn:

    """
 162 Create scatter plot between each feature and the response.
 163 - Plot title specifies feature name
 164 - Plot title specifies Pearson Correlation between feature and response
 165 - Plot saved under given folder with file name including feature name
 166 Parameters
 167 ----------
 168 X : DataFrame of shape (n_samples, n_features)
 169 Design matrix of regression problem
 170
 171 y : array-like of shape (n_samples, )
 172 Response vector to evaluate against
 173
 174 output_path: str (default ".")
 175 Path to folder in which plots are saved
 176 """

    sigma_y = np.std(y)

    for feature in X:
        for label in y:
            # tmp = y[label].unique()
            sigma_x = np.std(X[feature].astype(float))

            correlation_value = \
            ((np.cov(X[feature].astype(float), label)) / (sigma_x * sigma_y))[0, 1]

            fig = px.scatter(x=X[feature], y=y[label])

            fig.update_layout(
                title = f'The Correlation Between the Feature {feature} Values '
                f'with the Responses <br> Pearson Correlation v'
                    f'alues:{correlation_value}',

                xaxis_title = f'{feature} values',
                yaxis_title = 'Response values (y)')

                # fig.write_image(output_path + f"/pearson_value_for_{feature}.png")
            fig.show()#write_image(output_path + f"/pearson_value_for_{feature}.png")


