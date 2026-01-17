# ============================================================================ #
#                                                                              #
#     Title: Stability Algorithms                                              #
#     Purpose: Algorithms for measuring time series stability and lumpiness.   #
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
    This module provides algorithms to measure the stability and lumpiness of time series data.
"""


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Setup                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Imports                                                                   ####
# ---------------------------------------------------------------------------- #


# ## Python StdLib Imports ----
from typing import Union

# ## Python Third Party Imports ----
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tsfeatures import lumpiness as ts_lumpiness, stability as ts_stability
from typeguard import typechecked


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["stability", "lumpiness"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Algorithms                                                             ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# -----------------------------------------------------------------------------#
# Stability                                                                 ####
# -----------------------------------------------------------------------------#


@typechecked
def stability(data: Union[NDArray[np.float64], pd.DataFrame, pd.Series], freq: int = 1) -> float:
    r"""
    !!! note "Summary"
        Measure the stability of a time series by calculating the variance of the means across non-overlapping windows.

    ???+ abstract "Details"
        Stability is a feature extracted from time series data that quantifies how much the mean level of the series changes over time. It is particularly useful for identifying series with structural breaks or varying levels.

        The series is divided into non-overlapping "tiles" (windows) of length equal to the specified frequency. The mean of each tile is computed, and the stability is defined as the variance of these means. A higher value indicates lower stability (greater changes in the mean level).

    Params:
        data (Union[NDArray[np.float64], pd.DataFrame, pd.Series]):
            The time series data to analyse.
        freq (int):
            The number of observations per seasonal period or the desired window size for tiling.
            Default: `1`

    Returns:
        (float):
            The calculated stability value.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> import numpy as np
        >>> from ts_stat_tests.algorithms.stability import stability
        >>> from ts_stat_tests.utils.data import load_airline

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Measure stability of airline data"}
        >>> data = load_airline().values
        >>> res = stability(data, freq=12)
        >>> print(f"{res:.2f}")
        13428.67

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Measure stability of random noise"}
        >>> rng = np.random.RandomState(42)
        >>> data_random = rng.normal(0, 1, 144)
        >>> res = stability(data_random, freq=12)
        >>> print(f"{res:.4f}")
        0.0547

        ```

    ??? equation "Calculation"
        The stability $S$ is calculated by:

        1. Dividing the time series $X$ into $k$ non-overlapping windows $W_1, W_2, \dots, W_k$ of size $freq$.
        2. Computing the mean $\mu_i$ for each window $W_i$.
        3. Calculating the variance of these means:
        $$
        S = \text{Var}(\mu_1, \mu_2, \dots, \mu_k)
        $$

    ??? question "References"
        - Hyndman, R.J., Wang, X., & Laptev, N. (2015). Large-scale unusual time series detection. In Proceedings of the IEEE International Conference on Data Mining (ICDM 2015).

    ??? tip "See Also"
        - [`lumpiness()`][ts_stat_tests.algorithms.stability.lumpiness]
    """
    return ts_stability(x=data, freq=freq)["stability"]


# ------------------------------------------------------------------------------#
# Lumpiness                                                                  ####
# ------------------------------------------------------------------------------#


@typechecked
def lumpiness(data: Union[NDArray[np.float64], pd.DataFrame, pd.Series], freq: int = 1) -> float:
    r"""
    !!! note "Summary"
        Measure the lumpiness of a time series by calculating the variance of the variances across non-overlapping windows.

    ???+ abstract "Details"
        Lumpiness quantifies the extent to which the variance of a time series changes over time. It is useful for detecting series with "lumpy" patterns, where volatility is concentrated in certain periods.

        Similar to stability, the series is divided into non-overlapping tiles of length `freq`. Instead of means, the variance of each tile is computed. The lumpiness is defined as the variance of these tile variances. A higher value indicates greater "lumpiness" or inconsistent volatility.

    Params:
        data (Union[NDArray[np.float64], pd.DataFrame, pd.Series]):
            The time series data to analyse.
        freq (int):
            The number of observations per seasonal period or the desired window size for tiling.
            Default: `1`

    Returns:
        (float):
            The calculated lumpiness value.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> import numpy as np
        >>> from ts_stat_tests.algorithms.stability import lumpiness
        >>> from ts_stat_tests.utils.data import load_airline

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Measure lumpiness of airline data"}
        >>> data = load_airline().values
        >>> res = lumpiness(data, freq=12)
        >>> print(f"{res:.2f}")
        3986791.94

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Measure lumpiness of random noise"}
        >>> rng = np.random.RandomState(42)
        >>> data_random = rng.normal(0, 1, 144)
        >>> res = lumpiness(data_random, freq=12)
        >>> print(f"{res:.4f}")
        0.0925

        ```

    ??? equation "Calculation"
        The lumpiness $L$ is calculated by:

        1. Dividing the time series $X$ into $k$ non-overlapping windows $W_1, W_2, \dots, W_k$ of size $freq$.
        2. Computing the variance $\sigma^2_i$ for each window $W_i$.
        3. Calculating the variance of these variances:
        $$
        L = \text{Var}(\sigma^2_1, \sigma^2_2, \dots, \sigma^2_k)
        $$

    ??? question "References"
        - Hyndman, R.J., Wang, X., & Laptev, N. (2015). Large-scale unusual time series detection. In Proceedings of the IEEE International Conference on Data Mining (ICDM 2015).

    ??? tip "See Also"
        - [`stability()`][ts_stat_tests.algorithms.stability.stability]
    """
    return ts_lumpiness(x=data, freq=freq)["lumpiness"]
