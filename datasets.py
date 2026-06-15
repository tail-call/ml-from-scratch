from datetime import datetime

import numpy as np
import pandas as pd

IRIS_PATH = "data/iris/iris.data"
AIR_QUALITY_PATH = "data/air_quality/AirQualityUCI.csv"


def load_iris_dataset() -> np.ndarray:
    return np.loadtxt(
        IRIS_PATH,
        delimiter=",",
        converters={
            4: lambda x: {
                "Iris-setosa": 0,
                "Iris-versicolor": 1,
                "Iris-virginica": 2,
            }[x]
        },
    )


def convert_comma_float(x):
    return float(x.replace(",", "."))


def convert_date(x: str) -> float:
    """Parse '10/03/2024' → days since year 1."""
    return datetime.strptime(x, "%d/%m/%Y").toordinal()


def convert_time(x: str) -> float:
    """Parse '18.00.00' → fractional hours (0–24)."""
    return (
        datetime.strptime(x, "%H.%M.%S").hour
        + datetime.strptime(x, "%H.%M.%S").minute / 60
        + datetime.strptime(x, "%H.%M.%S").second / 3600
    )


def load_air_quality_dataset() -> np.ndarray:
    return np.loadtxt(
        AIR_QUALITY_PATH,
        delimiter=";",
        converters={
            0: convert_date,
            1: convert_time,
            2: convert_comma_float,
            5: convert_comma_float,
            12: convert_comma_float,
            13: convert_comma_float,
            14: convert_comma_float,
        },
        skiprows=1,
        usecols=range(15),
        max_rows=9357,
    )


def air_quality_dataset_column_names():
    with open(AIR_QUALITY_PATH) as file:
        line = file.readline()
        return line.split(";")[0:-2]


def air_quality_but_its_pandas() -> pd.DataFrame:
    # TODO: sine-cosine encoding of time (maybe even year?)
    df: pd.DataFrame = pd.read_csv(AIR_QUALITY_PATH, sep=";", decimal=",")
    df = df.drop(columns=["Date", "Time", "NMHC(GT)", "Unnamed: 15", "Unnamed: 16"])
    df = df.iloc[:9357]
    df = df.iloc[~(df == -200.0).any(axis=1)]
    return df
