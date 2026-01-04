# ============================================================================ #
#                                                                              #
#     Title: Data Utilities                                                    #
#     Purpose: Functions to load classic time series datasets.                 #
#                                                                              #
# ============================================================================ #


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Overview                                                              ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#  Description                                                              ####
# ---------------------------------------------------------------------------- #


"""
!!! note "Summary"
    This module contains utility functions to load classic time series datasets for testing and demonstration purposes.
"""


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Setup                                                                 ####
#                                                                              #
# ---------------------------------------------------------------------------- #


## --------------------------------------------------------------------------- #
##  Imports                                                                 ####
## --------------------------------------------------------------------------- #


# ## Python StdLib Imports ----
from functools import lru_cache

# ## Python Third Party Imports ----
import pandas as pd


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Data Loaders                                                          ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@lru_cache
def load_airline() -> pd.Series:
    """
    !!! note "Summary"
        Loads the classic Airline Passengers dataset as a pandas Series.

    Returns:
        (pd.Series):
            The Airline Passengers dataset.

    ??? success "Credit":
        Inspiration from: `sktime.datasets.load_airline()`

    ??? question "References":
        - Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis:
          forecasting and control. John Wiley & Sons.
    """
    data_source = "https://raw.githubusercontent.com/sktime/sktime/main/sktime/datasets/data/Airline/Airline.csv"
    _data = pd.read_csv(data_source, index_col=0, dtype={1: float}).squeeze("columns")
    if not isinstance(_data, pd.Series):
        raise TypeError("Expected a pandas Series from the data source.")
    data: pd.Series = _data
    data.index = pd.PeriodIndex(data.index, freq="M", name="Period")
    data.name = "Number of airline passengers"
    return data
