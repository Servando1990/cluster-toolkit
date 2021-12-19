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
#import psycopg2 # not yet in environment
warnings.filterwarnings("ignore")

TODO add more elegant ways to setup paths


def load_data(
    loaded_data,
    header=0,
    csv_conf={"separator": ",", "decimal": "."},
    table="pg_aggregate",
    datatype="",
    schema = "",
    user = "",
    password = "",
    host = "",
    port = "",
    database = ""
):

    """ Raw data load process
    Args:
        loaded_data = (str, pathlib.Path): Path or "sql" 
        header = (int): Row index used as column names
        csv_conf = (dict, optional): Define CSV separators
        table = (str, optional): If using Excel it means "sheet", if using json "key values", if sql means desired "table"
        datatype = optional. csv, json, xlsx
        user = (Postgres user database): Use just for sql tables. Default ""
        password = (Postgress password): Use just for sql tables. Default ""
        host = (Postgress host): Use just for sql tables. Default ""
        port = (Postgress port): Use just for sql tables. Default ""
        database = (Postgress database): Use just for sql tables. Default ""


    """
    # if str(loaded_data).lower() == 'sql':
    #     try:
    #         # conn = psycopg2.connect(user = user,password = password,
    #         # host = host,port = port , database = database)
    #         # cursor = conn.cursor()
    #         # sql = """select * from schema.table limit 100"""
    #         # cursor.execute(sql)
    #         # results = cursor.fetchall()
    #         # headers = [i[0] for i in cursor.description]
    #         # data =pd.DataFrame(results, columns = headers)

    #     except Exception:
    #         raise RuntimeError(logging.info("ERROR - Data load from SQL source failed -  "))

    #     return data

    data_path = Path(loaded_data)
    logging.info(loaded_data)

    try:
        if data_path.exists():
            loaded_data = Path(loaded_data).as_posix()
            file_path_exist = True
        else:
            raise FileNotFoundError

    except (FileNotFoundError, OSError):

        try:
            if data_path.exists():
                loaded_data = Path(loaded_data).as_posix()
                file_path_exist = True

            else:
                raise FileNotFoundError
        except (FileNotFoundError, OSError):
            file_path_exist = False

    #  take everything after last dot
    data_type_suffix = data_path.suffix[1:].lower()

    # # If not suffix inferred, then maybe url that return as request - than suffix have to be configured
    if not data_type_suffix or (
        data_type_suffix not in ["csv", "json", "xlsx"] and datatype
    ):
        data_type_suffix = datatype.lower()

    try:

        if data_type_suffix == "csv":

            if not header or header != 0:
                header = "infer"

            data = pd.read_csv(
                loaded_data,
                header=header,
                sep=csv_conf["separator"],
                decimal=csv_conf["decimal"],
            )

        elif data_type_suffix == "xlsx":
            data = pd.read_excel(loaded_data, sheet_name=table)

        elif data_type_suffix == "json":

            if file_path_exist:
                with open(loaded_data) as json_file:
                    data = (
                        json.load(json_file)[table] if table else json.load(json_file)
                    )

            else:
                data = (
                    json.loads(loaded_data)[table] if table else json.loads(loaded_data)
                )

        else:
            raise TypeError

    except TypeError:
        raise TypeError(
            logging.info(
                f"Your file format {data_type_suffix} not implemented yet. You can use csv, excel, json or txt.",
                "Wrong (not implemented) format",
            )
        )

    return data
