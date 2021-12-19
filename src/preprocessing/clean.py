
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn_pandas import DataFrameMapper
import os
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from feature_engine.encoding import CountFrequencyEncoder 

warnings.filterwarnings("ignore")

ID_COL = None
TARGET_VAR = None
RANDOM_SEED = 42



def preprocess_dataset(df):

    """
    Preprocessing stage for tabular data

    Parameters:
        ID_COL: Column ID of the dataframe
        TARGET: Target column if any 
    Args:
        df: Pandas Dataframe representation of tabular data to use

    Returns:
        scaled_features_df: Processed Dataframe
        mis_val_table_ren_columns: Dataframe with missing values detalied information
        
    """
    start = time.time()
    features = df.copy()

    # Extract the ids
    if ID_COL != None:
        features = df.drop(columns=[ID_COL])
    elif TARGET_VAR != None:
        features = df.drop(columns=[TARGET_VAR])

    le = LabelEncoder()
    le_count = 0

    for col in features:
        if features[col].dtype == "object":
            if len(list(features[col].unique())) <= 2:
                features[col] = le.fit_transform(
                    np.array(features[col].astype(str)).reshape((-1,))
                )
                le_count += 1
    print("%d columns were label encoded with LabelEncoder." % le_count)

    cutoff = 5

    features = pd.get_dummies(
        features,
        columns=features.columns[features.apply(pd.Series.nunique) <= cutoff],
        drop_first=True,
    )

    print("OneHotEncoding with a 5 dimension treshold")

    print(" Data Shape: ", features.shape)

    #  Handling missing values

    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: "Missing Values", 1: "% of Total Values"}
    )
    mis_val_table_ren_columns = (
        mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0]
        .sort_values("% of Total Values", ascending=False)
        .round(1)
    )

    print(
        "Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
        "There are "
        + str(mis_val_table_ren_columns.shape[0])
        + " columns that have missing values."
    )

    print("Handling missing values..")

    imputer = SimpleImputer(
        missing_values=np.nan, strategy="most_frequent"
    )  # mean only works for numeric data
    features[:] = imputer.fit_transform(features)

    for col in features:
        if features[col].dtype == "object":
            encoder = CountFrequencyEncoder(encoding_method="frequency")
            encoder.fit(features)
            features = encoder.transform(features)
            print("Encoding the rest of features with Categorical Frecuency")
        else:
            pass

    print("Bringing features onto the same scale")
    mapper = DataFrameMapper([(features.columns, StandardScaler())])
    scaled_features = mapper.fit_transform(features)
    scaled_features_df = pd.DataFrame(
        scaled_features, index=features.index, columns=features.columns
    )
    
    end = time.time()
    t = end - start 
    print("Your dataset has been processed succesfully, it took {} seconds". format(t))

    return scaled_features_df, mis_val_table_ren_columns




