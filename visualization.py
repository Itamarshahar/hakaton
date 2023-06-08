from pre_process import prepreprocess, run_preprocess, convert_to_dummies
import numpy as np
import plotly.express as px
import pandas as pd
from typing import NoReturn, Optional
import matplotlib.pyplot as plt

import matplotlib
def draw(X, y) -> None:
    """

    """
    print(X)
    print(y)
    # feature_evaluation(X,y)

    unique_labels = y[y.columns[0]].unique()
    for feature in X.columns:
        unique_feature = X[feature].unique()
        d = {str(num): sum(y[y.columns[0]==num]) for num in unique_feature}
        df = pd.DataFrame(d, index=unique_labels)
        ax = df.plot.bar(rot=0)
        ax.tick_params(axis='x', rotation=90, labelsize=5)
        ax.set_title(feature)  # Set feature name as the title
        plt.show()
"""
>>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
>>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
>>> index = ['snail', 'pig', 'elephant',
...          'rabbit', 'giraffe', 'coyote', 'horse']
>>> df = pd.DataFrame({'speed': speed,
...                    'lifespan': lifespan}, index=index)
>>> ax = df.plot.bar(rot=0)"""



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


