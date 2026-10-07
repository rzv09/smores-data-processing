"""
This file processes the 3OEC dataset https://www.bco-dmo.org/dataset/849934
"""

import pandas as pd
from datetime import datetime, timedelta

def make_timeseries(data_path: str):
    """
    Returns a dataframe object with proper time indexes

    Args:
        data_path (str): input data file location
    """
    df = pd.read_csv(data_path)
    df["O2_avg"] = df[["O2_S1", "O2_S2", "O2_S3"]].mean(axis=1)

    start_time_11 = datetime(2017, 7, 11, 14, 0, 0)
    end_time_11 = datetime(2017, 7, 12, 8, 0, 0)

    start_time_13 = datetime(2017, 7, 13, 11, 0, 0)
    end_time_13 = datetime(2017, 7, 14, 6, 0, 0)

    start_time_15 = datetime(2017, 7, 15, 10, 0, 0)
    end_time_15 = datetime(2017, 7, 16, 6, 0, 0)

    start_time_16 = datetime(2017, 7, 16, 16, 0, 0)
    end_time_16 = datetime(2017, 7, 17, 6, 0, 0)

    deployments = {
    "3oec_2017_7_11_12": {"start": start_time_11, "end": end_time_11},
    "3oec_2017_7_13_14": {"start": start_time_13, "end": end_time_13},
    "3oec_2017_7_15_16": {"start": start_time_15, "end": end_time_15},
    "3oec_2017_7_16_17": {"start": start_time_16, "end": end_time_16}
    }

    date_ranges = []

    for deployment_name, deployment_info in deployments.items():
        start_time = deployment_info["start"]
        end_time = deployment_info["end"]
        if deployment_name == "3oec_2017_7_13_14":
            start_time -= timedelta(seconds=0.125)
        print(start_time)

        # Calculate total seconds and number of measurements
        total_seconds = (end_time - start_time).total_seconds() + 0.125
        num_measurements = int(total_seconds * 8)

        # Create DatetimeIndex for the deployment
        date_range = pd.date_range(start=start_time, periods=num_measurements, freq=f'{1000/8}ms')
        print(date_range[0], date_range[-1])
        print(len(date_range))
        date_ranges.append(pd.Series(date_range))

    # Concatenate all DatetimeIndexes
    complete_index = pd.concat(date_ranges)

    # Set the complete index to your DataFrame
    df.index = complete_index

    return df

def get_first_piece(df: pd.DataFrame):
    """Returns first chronological piece of the dataframe"""
    return df["2017-07-11":"2017-07-12 06:00:00"]

def get_second_piece(df: pd.DataFrame):
    """Return second chronological piece of the dataframe"""
    return df["2017-07-13 12:00:00":"2017-07-14 06:00:00"]

def get_third_piece(df: pd.DataFrame):
    """Returns third chronological piece of the dataframe"""
    return df["2017-07-15 12:00:00":"2017-07-16 6:00:00"]

def get_fourth_piece(df: pd.DataFrame):
    """Returns fourth chronological piece of the dataframe"""
    return df["2017-07-16 16:00:00":"2017-07-17"]

def resample(df: pd.DataFrame, frequency: str, ):
    """
    Resample dataset using mean and frequency specified
    """
    df_resampled = df.drop(columns=['deployment', 't', 't_increase', 'Vx', 'Vy', 'Vz', 'P', 'O2_S1', 'O2_S2', 'O2_S3']).resample(frequency).mean()
    return df_resampled

def standardize_piece(df_piece):
    """
    Standardizes a chronological piece of the dataframe
      by subtracting the mean and dividing by std

    Args:
        df_piece (pandas dataframe)
    """

    # column_indices = {name: i for i, name in enumerate(df_piece.columns)}
    mean = df_piece.mean()
    std = df_piece.std()
    return (df_piece - mean) / std