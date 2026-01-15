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


## --------------------------------------------------------------------------- #
##  Description                                                             ####
## --------------------------------------------------------------------------- #


"""
!!! note "Summary"

    This module contains utility functions to load classic time series datasets for testing and demonstration purposes.

    It provides interfaces for both synthetic data generation (random numbers, sine waves, trends) and external data loading from common benchmarks.
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
from numpy.random import Generator as RandomGenerator
from numpy.typing import NDArray
from stochastic.processes.noise import FractionalGaussianNoise
from typeguard import typechecked


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
    "get_uniform_data",
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


SEED: int = 42


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Data Generators                                                       ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@lru_cache
@typechecked
def get_random_generator(seed: int) -> RandomGenerator:
    r"""
    !!! note "Summary"
        Generates a NumPy random number generator with a specified seed for reproducibility.

    ???+ abstract "Details"
        This function returns a `numpy.random.Generator` instance using `default_rng`. This is the recommended way to generate random numbers in modern NumPy (v1.17+).

    Params:
        seed (int):
            The seed value for the random number generator.

    Returns:
        (RandomGenerator):
            A NumPy random number generator initialized with the given seed.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.utils.data import get_random_generator

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Creating a RandomGenerator"}
        >>> rng = get_random_generator(42)
        >>> print(rng is not None)
        True
        >>> print(type(rng))
        <class 'numpy.random._generator.Generator'>

        ```

    ??? question "References"
        1. [NumPy Random Generator](https://numpy.org/doc/stable/reference/random/generator.html)

    """
    return np.random.default_rng(seed)


@lru_cache
@typechecked
def get_random_numbers(seed: int) -> NDArray[np.float64]:
    r"""
    !!! note "Summary"
        Generates an array of random numbers with a specified seed for reproducibility.

    ???+ abstract "Details"
        Generates a 1D array of 1000 random floating-point numbers distributed uniformly in the half-open interval $[0.0, 1.0)$.

    Params:
        seed (int):
            The seed value for the random number generator.

    Returns:
        (NDArray[np.float64]):
            An array of random numbers with shape (1000,).

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.utils.data import get_random_numbers
        >>> data = get_random_numbers(42)

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Generating Random Numbers"}
        >>> print(data.shape)
        (1000,)
        >>> print(type(data))
        <class 'numpy.ndarray'>
        >>> print(data.dtype)
        float64
        >>> print(data[:5])
        [0.77395605 0.43887844 0.85859792 0.69736803 0.09417735]

        ```

    ??? question "References"
        1. [NumPy Random Generator](https://numpy.org/doc/stable/reference/random/generator.html)

    """
    rng: RandomGenerator = get_random_generator(seed)
    return rng.random(size=1000)


@lru_cache
@typechecked
def get_random_numbers_2d(seed: int) -> NDArray[np.float64]:
    r"""
    !!! note "Summary"
        Generates a 2D array of random numbers with a specified seed for reproducibility.

    ???+ abstract "Details"
        Produces a 2D matrix of shape $(4, 3000)$ containing uniform random values.

    Params:
        seed (int):
            The seed value for the random number generator.

    Returns:
        (NDArray[np.float64]):
            A 2D array of random numbers with shape (4, 3000).

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.utils.data import get_random_numbers_2d
        >>> data = get_random_numbers_2d(42)

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Generating 2D Random Numbers"}
        >>> print(data.shape)
        (4, 3000)
        >>> print(type(data))
        <class 'numpy.ndarray'>
        >>> print(data.dtype)
        float64
        >>> print(data[:, :5])
        [[0.06206311 0.45826204 0.12903006 0.15232671 0.63228281]
         [0.71609997 0.3571156  0.85186786 0.24097716 0.53839349]
         [0.74315144 0.90157433 0.59866347 0.52857443 0.89016256]
         [0.72072839 0.71123776 0.20269503 0.0366554  0.30379952]]

        ```

    ??? question "References"
        1. [NumPy Random Generator](https://numpy.org/doc/stable/reference/random/generator.html)

    """
    rng: RandomGenerator = get_random_generator(seed)
    return rng.random(size=(4, 3000))


@lru_cache
@typechecked
def get_sine_wave() -> NDArray[np.float64]:
    r"""
    !!! note "Summary"
        Generates a sine wave dataset.

    ???+ abstract "Details"
        Produces a 1D array of 1000 samples of a sine wave with amplitude 1.0 and period 1000.

    Returns:
        (NDArray[np.float64]):
            An array representing a sine wave with shape (3000,).

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.utils.data import get_sine_wave
        >>> data = get_sine_wave()

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Generating a Sine Wave"}
        >>> print(data.shape)
        (3000,)
        >>> print(type(data))
        <class 'numpy.ndarray'>
        >>> print(data.dtype)
        float64
        >>> print(data[:5])
        [0.         0.06279052 0.12533323 0.18738131 0.24868989]

        ```

    ??? question "References"
        1. [NumPy Trigonometric Functions](https://numpy.org/doc/stable/reference/routines.math.html#trigonometric-functions)

    """
    return np.sin(2 * np.pi * 1 * np.arange(3000) / 100)


@lru_cache
@typechecked
def get_normal_curve(seed: int) -> NDArray[np.float64]:
    r"""
    !!! note "Summary"
        Generates a normal distribution curve dataset.

    ???+ abstract "Details"
        Draws 1000 samples from a standard Gaussian distribution.

    Params:
        seed (int):
            The seed value for the random number generator.

    Returns:
        (NDArray[np.float64]):
            An array representing a normal distribution curve with shape (1000,).

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.utils.data import get_normal_curve
        >>> data = get_normal_curve(42)

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Generating Normal Curve Data"}
        >>> print(data.shape)
        (1000,)
        >>> print(type(data))
        <class 'numpy.ndarray'>
        >>> print(data.dtype)
        float64
        >>> print(data[:5])
        [ 0.12993113 -0.75691222 -0.33007356 -1.88579735 -0.37064992]

        ```

    ??? question "References"
        1. [NumPy Random Normal](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.normal.html)

    """
    rng: RandomGenerator = get_random_generator(seed)
    return rng.normal(loc=0.0, scale=1.0, size=1000)


@lru_cache
@typechecked
def get_straight_line() -> NDArray[np.float64]:
    r"""
    !!! note "Summary"
        Generates a straight line dataset.

    ???+ abstract "Details"
        Returns a sequence of integers from 0 to 999.

    Returns:
        (NDArray[np.float64]):
            An array representing a straight line with shape (1000,).

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.utils.data import get_straight_line
        >>> data = get_straight_line()

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Generating Straight Line Data"}
        >>> print(data.shape)
        (1000,)
        >>> print(type(data))
        <class 'numpy.ndarray'>
        >>> print(data.dtype)
        float64
        >>> print(data[:5])
        [0. 1. 2. 3. 4.]

        ```

    ??? question "References"
        1. [NumPy Arange](https://numpy.org/doc/stable/reference/generated/numpy.arange.html)

    """
    return np.arange(1000).astype(np.float64)


@lru_cache
@typechecked
def get_trend_data() -> NDArray[np.float64]:
    r"""
    !!! note "Summary"
        Generates trend data.

    ???+ abstract "Details"
        Generates an array with a linear trend by combining two ramp functions.

    Returns:
        (NDArray[np.float64]):
            An array representing trend data with shape (1000,).

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.utils.data import get_trend_data
        >>> data = get_trend_data()

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Generating Trend Data"}
        >>> print(data.shape)
        (1000,)
        >>> print(type(data))
        <class 'numpy.ndarray'>
        >>> print(data.dtype)
        float64
        >>> print(data[:5])
        [0.  1.5 3.  4.5 6. ]

        ```

    ??? question "References"
        1. [NumPy Arange](https://numpy.org/doc/stable/reference/generated/numpy.arange.html)

    """
    return np.arange(1000) + 0.5 * np.arange(1000)


@lru_cache
@typechecked
def get_uniform_data(seed: int) -> NDArray[np.float64]:
    r"""
    !!! note "Summary"
        Generates uniform random data with a specified seed for reproducibility.

    ???+ abstract "Details"
        Returns a 1D array of 1000 samples from a uniform distribution.

    Params:
        seed (int):
            The seed value for the random number generator.

    Returns:
        (NDArray[np.float64]):
            An array of uniform random data with shape (1000,).

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.utils.data import get_uniform_data
        >>> data = get_uniform_data(42)

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Generating Uniform Data"}
        >>> print(data.shape)
        (1000,)
        >>> print(type(data))
        <class 'numpy.ndarray'>
        >>> print(data.dtype)
        float64
        >>> print(data[:5])
        [0.80227457 0.81857128 0.87962986 0.11378193 0.29263938]

        ```

    ??? question "References"
        1. [NumPy Random Uniform](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.uniform.html)

    """
    rng: RandomGenerator = get_random_generator(seed)
    return rng.uniform(low=0.0, high=1.0, size=1000)


@lru_cache
@typechecked
def get_noise_data(seed: int) -> NDArray[np.float64]:
    r"""
    !!! note "Summary"
        Generates fractional Gaussian noise data with a specified seed for reproducibility.

    ???+ abstract "Details"
        Uses the `stochastic` library to sample fractional Gaussian noise with a Hurst exponent of 0.5, effectively producing white noise.

    Params:
        seed (int):
            The seed value for the random number generator.

    Returns:
        (NDArray[np.float64]):
            An array of fractional Gaussian noise data with shape (1000,).

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.utils.data import get_noise_data
        >>> data = get_noise_data(42)

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Generating Noise Data"}
        >>> print(data.shape)
        (1000,)
        >>> print(type(data))
        <class 'numpy.ndarray'>
        >>> print(data.dtype)
        float64
        >>> print(data[:5])
        [-0.05413957 -0.0007609  -0.00177524  0.00909899 -0.03044404]

        ```

    ??? question "References"
        1. [Stochastic Library](https://github.com/crflynn/stochastic)

    """
    rng: RandomGenerator = get_random_generator(seed)
    return FractionalGaussianNoise(hurst=0.5, rng=rng).sample(1000)


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Data Loaders                                                          ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@lru_cache
@typechecked
def load_airline() -> pd.Series:
    r"""
    !!! note "Summary"
        Loads the classic Airline Passengers dataset as a pandas Series.

    ???+ abstract "Details"
        The Air Passengers dataset provides monthly totals of a US airline's international passengers from 1949 to 1960. It is a classic dataset for time series analysis, exhibiting both trend and seasonality.

    Returns:
        (pd.Series):
            The Airline Passengers dataset.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.utils.data import load_airline
        >>> data = load_airline()

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Loading Airline Data"}
        >>> print(len(data))
        144
        >>> print(type(data))
        <class 'pandas.core.series.Series'>
        >>> print(data.head())
        Period
        1949-01    112.0
        1949-02    118.0
        1949-03    132.0
        1949-04    129.0
        1949-05    121.0
        Freq: M, Name: Number of airline passengers, dtype: float64

        ```

    ??? success "Credit"
        Inspiration from: [`sktime.datasets.load_airline()`](https://www.sktime.net/en/stable/api_reference/generated/sktime.datasets.load_airline.html)

    ??? question "References"
        1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: forecasting and control. John Wiley & Sons.

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
@typechecked
def load_macrodata() -> pd.DataFrame:
    r"""
    !!! note "Summary"
        Loads the classic Macrodata dataset as a pandas DataFrame.

    ???+ abstract "Details"
        This dataset contains various US macroeconomic time series from 1959Q1 to 2009Q3. It includes variables such as real GDP, consumption, investment, etc.

    Returns:
        (pd.DataFrame):
            The Macrodata dataset.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.utils.data import load_macrodata
        >>> data = load_macrodata()

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Loading Macrodata"}
        >>> print(data.shape)
        (203, 14)
        >>> print(type(data))
        <class 'pandas.core.frame.DataFrame'>
        >>> print(data[["year", "quarter", "realgdp"]].head())
                year  quarter   realgdp
        Period
        1959Q1  1959        1  2710.349
        1959Q2  1959        2  2778.801
        1959Q3  1959        3  2775.488
        1959Q4  1959        4  2785.204
        1960Q1  1960        1  2847.699

        ```

    ??? success "Credit"
        Inspiration from: [`statsmodels.datasets.macrodata.load_pandas()`](https://www.statsmodels.org/stable/datasets/generated/statsmodels.datasets.macrodata.macrodata.load_pandas.html)

    ??? question "References"
        1. R. F. Engle, D. F. Hendry, and J. F. Richard (1983). Exogeneity. Econometrica, 51(2):277–304.

    """
    data_source = (
        "https://raw.githubusercontent.com/statsmodels/statsmodels/main/statsmodels/datasets/macrodata/macrodata.csv"
    )
    data: pd.DataFrame = pd.read_csv(
        data_source,
        index_col=None,
        dtype={
            "year": int,
            "quarter": int,
        },
    )
    data.index = pd.PeriodIndex(
        data=data.year.astype(str) + "Q" + data.quarter.astype(str),
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
data_random: NDArray[np.float64] = get_random_numbers(SEED)
data_random_2d: NDArray[np.float64] = get_random_numbers_2d(SEED)
data_sine: NDArray[np.float64] = get_sine_wave()
data_normal: NDArray[np.float64] = get_normal_curve(SEED)
data_line: NDArray[np.float64] = get_straight_line()
data_trend: NDArray[np.float64] = get_trend_data()
data_noise: NDArray[np.float64] = get_noise_data(SEED)
