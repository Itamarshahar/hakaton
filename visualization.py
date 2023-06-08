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

def catagorial_label_perc_():



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


