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
import numpy as np
import pandas as pd
from numpy.random._generator import Generator as RandomGenerator
from stochastic.processes.noise import FractionalGaussianNoise


## --------------------------------------------------------------------------- #
##  Exports                                                                 ####
## --------------------------------------------------------------------------- #


__all__: list[str] = [
    "get_random_generator",
    "get_random_numbers",
    "get_random_numbers_2d",
    "get_sine_wave",
    "get_normal_curve",
    "get_straight_line",
    "get_trend_data",
    "get_noise_data",
    "load_airline",
    "load_macrodata",
    "data_airline",
    "data_macrodata",
    "data_random",
    "data_random_2d",
    "data_sine",
    "data_normal",
    "data_line",
    "data_trend",
    "data_noise",
]


## --------------------------------------------------------------------------- #
##  Constants                                                               ####
## --------------------------------------------------------------------------- #


SEED = 42


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Data Generators                                                       ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@lru_cache
def get_random_generator(seed: int) -> RandomGenerator:
    """
    !!! note "Summary"
        Generates a NumPy random number generator with a specified seed for reproducibility.

    Params:
        seed (int):
            The seed value for the random number generator.

    Returns:
        (RandomGenerator):
            A NumPy random number generator initialized with the given seed.
    """
    return np.random.default_rng(seed)


@lru_cache
def get_random_numbers(seed: int) -> np.ndarray:
    """
    !!! note "Summary"
        Generates an array of random numbers with a specified seed for reproducibility.

    Params:
        seed (int):
            The seed value for the random number generator.

    Returns:
        (np.ndarray):
            An array of random numbers with shape (1000,).
    """
    rng: RandomGenerator = get_random_generator(seed)
    return rng.random(size=1000)


@lru_cache
def get_random_numbers_2d(seed: int) -> np.ndarray:
    """
    !!! note "Summary"
        Generates a 2D array of random numbers with a specified seed for reproducibility.

    Params:
        seed (int):
            The seed value for the random number generator.

    Returns:
        (np.ndarray):
            A 2D array of random numbers with shape (4, 3000).
    """
    rng: RandomGenerator = get_random_generator(seed)
    return rng.random(size=(4, 3000))


@lru_cache
def get_sine_wave() -> np.ndarray:
    """
    !!! note "Summary"
        Generates a sine wave dataset.

    Returns:
        (np.ndarray):
            An array representing a sine wave with shape (3000,).
    """
    return np.sin(2 * np.pi * 1 * np.arange(3000) / 100)


@lru_cache
def get_normal_curve(seed: int) -> np.ndarray:
    """
    !!! note "Summary"
        Generates a normal distribution curve dataset.

    Params:
        seed (int):
            The seed value for the random number generator.

    Returns:
        (np.ndarray):
            An array representing a normal distribution curve with shape (1000,).
    """
    rng: RandomGenerator = get_random_generator(seed)
    return rng.normal(loc=0.0, scale=1.0, size=1000)


@lru_cache
def get_straight_line() -> np.ndarray:
    """
    !!! note "Summary"
        Generates a straight line dataset.

    Returns:
        (np.ndarray):
            An array representing a straight line with shape (1000,).
    """
    return np.arange(1000)


@lru_cache
def get_trend_data() -> np.ndarray:
    """
    !!! note "Summary"
        Generates trend data.

    Returns:
        (np.ndarray):
            An array representing trend data with shape (1000,).
    """
    return np.arange(1000) + 0.5 * np.arange(1000)


@lru_cache
def get_uniform_data(seed: int) -> np.ndarray:
    """
    !!! note "Summary"
        Generates uniform random data with a specified seed for reproducibility.

    Params:
        seed (int):
            The seed value for the random number generator.

    Returns:
        (np.ndarray):
            An array of uniform random data with shape (1000,).
    """
    rng: RandomGenerator = get_random_generator(seed)
    return rng.uniform(low=0.0, high=1.0, size=1000)


@lru_cache
def get_noise_data(seed: int) -> np.ndarray:
    """
    !!! note "Summary"
        Generates fractional Gaussian noise data with a specified seed for reproducibility.

    Params:
        seed (int):
            The seed value for the random number generator.

    Returns:
        (np.ndarray):
            An array of fractional Gaussian noise data with shape (1000,).
    """
    rng: RandomGenerator = get_random_generator(seed)
    return FractionalGaussianNoise(hurst=0.5, rng=rng).sample(1000)


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

    !!! success "Credit"
        Inspiration from: [`sktime.datasets.load_airline()`](https://www.sktime.net/en/stable/api_reference/generated/sktime.datasets.load_airline.html)

    ??? question "References"
        - Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: forecasting and control. John Wiley & Sons.
    """
    data_source = "https://raw.githubusercontent.com/sktime/sktime/main/sktime/datasets/data/Airline/Airline.csv"
    _data = pd.read_csv(data_source, index_col=0, dtype={1: float}).squeeze("columns")
    if not isinstance(_data, pd.Series):
        raise TypeError("Expected a pandas Series from the data source.")
    data: pd.Series = _data
    data.index = pd.PeriodIndex(data.index, freq="M", name="Period")
    data.name = "Number of airline passengers"
    return data


@lru_cache
def load_macrodata() -> pd.DataFrame:
    """
    !!! note "Summary"
        Loads the classic Macrodata dataset as a pandas DataFrame.

    Returns:
        (pd.DataFrame):
            The Macrodata dataset.

    !!! success "Credit"
        Inspiration from: [`statsmodels.datasets.macrodata.load_pandas()`](https://www.statsmodels.org/stable/datasets/generated/statsmodels.datasets.macrodata.macrodata.load_pandas.html)

    ??? question "References"
        - R. F. Engle, D. F. Hendry, and J. F. Richard (1983). Exogeneity. Econometrica, 51(2):277–304.
    """
    data_source = (
        "https://raw.githubusercontent.com/statsmodels/statsmodels/main/statsmodels/datasets/macrodata/macrodata.csv"
    )
    _data = pd.read_csv(data_source, index_col=None, dtype={1: float})
    if not isinstance(_data, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame from the data source.")
    data: pd.DataFrame = _data
    data.index = data.year
    data.index = pd.PeriodIndex(
        data=data.index,
        freq="Q",
        name="Period",
    )
    return data


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Data Objects                                                          ####
#                                                                              #
# ---------------------------------------------------------------------------- #


data_airline: pd.Series = load_airline()
data_macrodata: pd.DataFrame = load_macrodata()
data_random: np.ndarray = get_random_numbers(SEED)
data_random_2d: np.ndarray = get_random_numbers_2d(SEED)
data_sine: np.ndarray = get_sine_wave()
data_normal: np.ndarray = get_normal_curve(SEED)
data_line: np.ndarray = get_straight_line()
data_trend: np.ndarray = get_trend_data()
data_noise: np.ndarray = get_noise_data(SEED)
