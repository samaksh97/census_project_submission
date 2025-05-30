import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    clf = DecisionTreeClassifier(random_state=42)
    cv = KFold(5, shuffle=True, random_state=42)
    trainer = GridSearchCV(clf,
                           {"max_depth": np.linspace(5, 30, 6).astype(int)},
                           cv=cv)
    trainer.fit(X_train, y_train)
    bst_model = trainer.best_estimator_
    return bst_model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.  # NOQA:E501

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return {"precision": precision,
            "recall": recall,
            "fbeta": fbeta}


def compute_metrics_on_slice(data, y_pred, y_true, feature_name):
    """evaluation model performance on specific features, return
    a dict of performance dict, that key is attribute value in the
    features.

    Args:
        data: A pandas DataFrame to be analysised by model
        y_pred: 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
        y_true: 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
        model: A trained model
        feature_name: the feature name of data
    """
    _tmp_data = data
    _tmp_data['label'] = y_true
    _tmp_data['pred'] = y_pred

    ans = {}
    for attr in data[feature_name].unique():
        _slice = data[data[feature_name] == attr]
        perf = compute_model_metrics(y=_slice['label'],
                                     preds=_slice['pred'])
        ans[attr] = perf

    return ans


def plot_model_disparity_on_fpr(data: pd.DataFrame, output_path: str):
    """calculate model's disparity on each catgorical features.

    Args:
        data (pd.DataFrame): a pandas dataframe with catgorical features and model's predict result  # NOQA:E501
        output (str): a path of folder to save disparity on graph
    """
    from aequitas.group import Group
    from aequitas.plotting import Plot
    from aequitas.preprocessing import preprocess_input_df
    df, _ = preprocess_input_df(data)
    g = Group()
    aqp = Plot()
    xtab, _ = g.get_crosstabs(df)

    figure, ax = plt.subplots(1, 1, figsize=(12, 32))
    _ = aqp.plot_group_metric(xtab, 'fpr', ax=ax)
    figure.savefig(os.path.join(output_path, "fpr_fiarness_graph.png"))


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds
