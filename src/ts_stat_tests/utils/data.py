# ## Python Third Party Imports ----
import numpy as np
import pandas as pd


def load_airline() -> pd.Series:
    # Inspiration from: sktime.datasets.load_airline()
    data_source = "https://raw.githubusercontent.com/sktime/sktime/main/sktime/datasets/data/Airline/Airline.csv"
    data: pd.Series = pd.read_csv(data_source, index_col=0, dtype={1: float}).squeeze("columns")
    data.index = pd.PeriodIndex(data.index, freq="M", name="Period")
    data.name = "Number of airline passengers"
    return data
