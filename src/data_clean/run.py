import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

ZERO_RATE = 0.95

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def data_cleaning(input_path: str, output_path: str):
    """a simple data cleaning pipeline

    Args:
        input_path (str): a local path store the data
        output_path (str): a local path output the cleaned data
    """
    # find all csv file under folder
    comman_run_path = Path(os.getcwd()).parent.parent
    input_path = os.path.join(str(comman_run_path), input_path)
    output_path = os.path.join(str(comman_run_path), output_path)

    files_wait_ingested = [str(f) for f in Path(input_path).glob('*.csv')]
    data = pd.DataFrame()
    for csv in Path(input_path).glob('*.csv'):
        _df = pd.read_csv(os.path.join(input_path, csv))
        if data is None:
            data = _df
        else:
            data = pd.concat([data, _df], axis=0)
    # normalize data columns name
    logger.info('STEP[data_clean]: (1/3) Begin...')
    columns = [col.lstrip() for col in data.columns]
    data.columns = columns
    # remove duplicate rows
    data = data.drop_duplicates()
    # remove features with high zero rate %
    drop_col = []
    for col in data.columns:
        if is_numeric_dtype(data[col]):
            # check zero rate
            zero_rate = len(data[data[col] == 0])/len(data)
            if zero_rate > ZERO_RATE:
                drop_col.append(col)

    data = data.drop(columns=drop_col)
    data = data.dropna(axis=0)

    logger.info('STEP[data_clean]: (2/3) Clean Completed')

    # write a data ingestion log to the output path
    with open(os.path.join(output_path, 'data_clean_log.txt'), 'w+') as f:
        time = datetime.now().strftime('%Y-%m-%d-%H-%m-%s')
        f.write(time + '\n')
        for file in files_wait_ingested:
            f.write(file + "\n")
    data.to_csv(os.path.join(output_path, 'cleaned_data.csv'), index=False)
    logger.info('STEP[data_clean]: (3/3) Output Completed')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    parser.add_argument("--input_path",
                        type=str,
                        help="path of the raw data stored",
                        required=True)

    parser.add_argument("--output_path",
                        type=str,
                        help="path of the data need to be output",
                        required=True)

    args = parser.parse_args()

    data_cleaning(args.input_path, args.output_path)
