import argparse
import json
import logging
import os
import pickle
from pathlib import Path

import pandas as pd
from ml.data import cat_features, process_data
from ml.model import (compute_metrics_on_slice, compute_model_metrics,
                      plot_model_disparity_on_fpr, train_model)
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def load_data(file_path: str):
    """this function load data and split it into train, test

    Args:
        file_path (str): the path of data
    """
    data = pd.read_csv(file_path)
    return data


def training(input_path: str, output_path: str):
    runtime_path = Path(os.getcwd()).parent.parent
    input_path = os.path.join(runtime_path, input_path)
    output_path = os.path.join(runtime_path, output_path)

    data = load_data(input_path)
    logger.info("STEP[train]: (1/3) Data Loaded.")
    # generate k-fold

    # train, test split
    train_data, test_data = train_test_split(data, test_size=0.2,
                                             random_state=42)

    X_train, y_train, encoder, lb = process_data(train_data, categorical_features=cat_features,  # NOQA:E501
                                                 label="salary", training=True)
    with open(os.path.join(output_path, 'encoder.pkl'), 'wb') as mf:
        pickle.dump(encoder, mf)
    # save model and score
    model = train_model(X_train, y_train)
    logger.info("STEP[train]: (2/3) Train Completed.")
    # eval on test dataset
    X_test, y_test, _, _ = process_data(test_data, categorical_features=cat_features,  # NOQA:E501
                                        label='salary', training=False,
                                        encoder=encoder, lb=lb)
    y_pred = model.predict(X_test)
    scores = compute_model_metrics(y_test, y_pred)
    # evaluation on data slice
    perf_on_slice = compute_metrics_on_slice(data=test_data,
                                             y_pred=y_pred, y_true=y_test,
                                             feature_name='education')

    with open(os.path.join(output_path, 'slice_output.txt'), 'w+') as f:
        json.dump(perf_on_slice, f)
    #
    test_data = test_data.rename(columns={'salary': 'label_value'})
    test_data['label_value'] = y_test
    test_data['score'] = y_pred

    plot_model_disparity_on_fpr(test_data[cat_features + ['score', 'label_value']],  # NOQA: E501
                                output_path=output_path)

    logger.info("STEP[train]: (3/3) Eval Completed.")
    # save model and score
    with open(os.path.join(output_path, 'dct_model.pkl'), 'wb') as mf:
        pickle.dump(model, mf)

    with open(os.path.join(output_path, 'model_score.txt'), "w") as f:
        f.write(f"Label Encoder: {lb.classes_}\n")
        for eval_metric, metric_score in scores.items():
            f.write(f"{eval_metric}: {metric_score:.3f}\n")
    logger.info("STEP[train]: Final Result Saved.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    parser.add_argument("--input_path",
                        type=str,
                        help="path of the raw data stored",
                        required=True)

    parser.add_argument("--output_path",
                        type=str,
                        help="path of the model need to be output",
                        required=True)

    args = parser.parse_args()

    training(args.input_path, args.output_path)
