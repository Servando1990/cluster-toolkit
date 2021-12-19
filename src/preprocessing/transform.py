import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import urllib
import requests
import itertools
from sklearn import preprocessing
import json
#from mylogging import user_warning, user_message
import logging
import warnings

#logging.warning('Watch out!')  # will print a message to the console
#logging.info('I told you so')  # will not print anything

warnings.filterwarnings("ignore")


def transform_data(
    data,
    data_orientation="columns",
):
    """Transform input data from raw loading to a pd.Dataframe ready to preprocess.

    Args:
        data (np.ndarray, pd.DataFrame, list or dict): Input data in well standardized format.
        data_orientation (str, optional): 'columns' or 'index'.
    Raises:
        KeyError, TypeError: If wrong configuration in inut format.

    Returns:
        np.ndarray, pd.DataFrame: df ready for preprocess

    """

    if isinstance(data, np.ndarray):

        data = pd.DataFrame(data)

    elif isinstance(data, list):
        data = pd.DataFrame.from_records(data)

    elif isinstance(data, dict):

        # If just one column, put in list to have same syntax further
        if not isinstance(next(iter(data.values())), list):
            data = {i: [j] for (i, j) in data.items()}

        # orientation = "columns" if not data_orientation else data_orientation
        data = pd.DataFrame.from_dict(data, orient=data_orientation)

    else:
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except Exception as err:
                raise (
                    RuntimeError(
                        logging.info(
                            "Supported formats.(np.ndarray, list, dict, pd.DataFrame))"
                            f"\n\n Detailed error: \n\n {err}",
                            caption="Data load failed",
                        )
                    )
                )

    if isinstance(data, pd.DataFrame):

        df_ready = data.copy()
        # Remove list and dicts from json input in Pandas columns
        a = df_ready.applymap(lambda x: isinstance(x, list)).all()
        L = a.index[a].tolist()
        df_ready.drop(L, axis=1, inplace=True)

    else:
        raise TypeError(
            logging.info(
                "Input data must be in pd.dataframe, pd.series, numpy array or in a path (str or pathlib) with supported formats"
                " - csv, xlsx, txt ",
                "Data format error",
            )
        )

    return df_ready
