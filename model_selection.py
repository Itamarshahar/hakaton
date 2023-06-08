import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import plotly.graph_objects as go

def run_model_selection(X_train, X_test, y_train, y_test):
    k_range = list(range(1, 40, 2))


    train_errors, test_errors = [], []
    for k in k_range:
        model = KNeighborsClassifier(k).fit(X_train, y_train)
        train_errors.append(1 - model.score(X_train, y_train))
        test_errors.append(1 - model.score(X_test, y_test))

    param_grid = {'n_neighbors': k_range}
    knn_cv = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3).fit(X_train,
                                                                        y_train)
    cv_errors = 1 - knn_cv.cv_results_["mean_test_score"]
    std = knn_cv.cv_results_["std_test_score"]

    min_ind = np.argmin(np.array(cv_errors))
    selected_k = np.array(k_range)[min_ind]
    selected_error = cv_errors[min_ind]

    go.Figure([
        go.Scatter(name='Lower CV Error CI', x=k_range, y=cv_errors - 2 * std,
                   mode='lines', line=dict(color="lightgrey"), showlegend=False,
                   fill=None),
        go.Scatter(name='Upper CV Error CI', x=k_range, y=cv_errors + 2 * std,
                   mode='lines', line=dict(color="lightgrey"), showlegend=False,
                   fill="tonexty"),

        go.Scatter(name="Train Error", x=k_range, y=train_errors,
                   mode='markers + lines', marker_color='rgb(152,171,150)'),
        go.Scatter(name="CV Error", x=k_range, y=cv_errors, mode='markers + lines',
                   marker_color='rgb(220,179,144)'),
        go.Scatter(name="Test Error", x=k_range, y=test_errors,
                   mode='markers + lines', marker_color='rgb(25,115,132)'),
        go.Scatter(name='Selected Model', x=[selected_k], y=[selected_error],
                   mode='markers',
                   marker=dict(color='darkred', symbol="x", size=10))]) \
        .update_layout(
        title=r"$\text{(4) }k\text{-NN Errors - Selection By Cross-Validation}$",
        xaxis_title=r"$k\text{ - Number of Neighbors}$",
        yaxis_title=r"$\text{Error Value}$").show()