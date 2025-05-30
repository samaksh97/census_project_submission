import math

import numpy as np
import pytest
from sklearn.base import is_classifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression

from .ml.model import compute_model_metrics, inference, train_model


@pytest.mark.parametrize('y_true, y_pred',
                         [([0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 1])])
def test_compute_model_metrics(y_true: list, y_pred: list):
    ans = compute_model_metrics(y=y_true,
                                preds=y_pred)
    prec, recall, fbeta = ans['precision'], ans['recall'], ans['fbeta']

    assert math.isclose(prec, 0.5, abs_tol=0.1), "calculate falut in precision"
    assert math.isclose(recall, 0.666666, abs_tol=0.1), "calculate falut in recall"  # NOQA:E501
    assert math.isclose(fbeta, 0.5714, abs_tol=0.1), "calculate falut in fbeta"


def test_inference():
    model = LinearRegression()
    X_train = np.array([[1], [2], [3], [4]])
    y_train = np.array([2, 3, 4, 5])
    model.fit(X_train, y_train)

    # Create a sample dataset for testing
    X_test = np.array([[5], [6]])
    expected_output = np.array([6, 7])  # Expected output

    # Run the inference function
    preds = inference(model, X_test)
    np.testing.assert_array_almost_equal(preds, expected_output, decimal=1)


def test_train_model():
    X_train, y_train = make_classification(n_samples=100, n_features=4,
                                           random_state=42)
    # Train the model
    model = train_model(X_train, y_train)

    # Check if the returned model is a trained classifier
    assert is_classifier(model), "return Value is not Classifier"
    # Optionally, check specific attributes (e.g., max_depth)
    assert model.max_depth in list(range(5, 31)), "return Model depth fault"
