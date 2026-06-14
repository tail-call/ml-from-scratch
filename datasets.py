from datetime import datetime

import numpy as np


def load_iris_dataset() -> np.ndarray:
    return np.loadtxt(
        "data/iris/iris.data",
        delimiter=",",
        converters={
            4: lambda x: {
                "Iris-setosa": 0,
                "Iris-versicolor": 1,
                "Iris-virginica": 2,
            }[x]
        },
    )


def load_air_quality_dataset() -> np.ndarray:
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

    return np.loadtxt(
        "data/air_quality/AirQualityUCI.csv",
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
