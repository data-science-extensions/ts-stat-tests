# ============================================================================ #
#                                                                              #
#     Title: Stability Tests                                                   #
#     Purpose: Convenience functions for stability algorithms.                 #
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
    This module contains convenience functions and tests for stability measures,
    allowing for easy access to stability and lumpiness algorithms.
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
from typeguard import typechecked

# ## Local First Party Imports ----
from ts_stat_tests.algorithms.stability import (
    lumpiness as _lumpiness,
    stability as _stability,
)


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["is_stable", "is_lumpy"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Algorithms                                                             ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# -----------------------------------------------------------------------------#
# Stability                                                                 ####
# -----------------------------------------------------------------------------#


@typechecked
def is_stable(
    data: Union[NDArray[np.float64], pd.DataFrame, pd.Series],
    freq: int = 1,
    alpha: float = 0.5,
) -> bool:
    r"""
    !!! note "Summary"
        Determine if a time series is stable based on a variance threshold.

    ???+ abstract "Details"
        A time series is considered stable if the variance of its windowed means (the stability metric) is below a specified threshold (`alpha`). High stability values indicate that the mean level of the series changes significantly over time (e.g., due to trends or structural breaks).

    Params:
        data (Union[NDArray[np.float64], pd.DataFrame, pd.Series]):
            The time series data to analyse.
        freq (int):
            The number of observations per seasonal period or the desired window size for tiling.
            Default: `1`
        alpha (float):
            The threshold for stability. The series is considered stable if the calculated stability metric is less than this value.
            Default: `0.5`

    Returns:
        (bool):
            `True` if the stability metric is less than `alpha` (the series is stable), otherwise `False`.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> import numpy as np
        >>> from ts_stat_tests.tests.stability import is_stable
        >>> from ts_stat_tests.utils.data import load_airline

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Check if airline data is stable"}
        >>> data = load_airline().values
        >>> is_stable(data, freq=12, alpha=1.0)
        False

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Check if random noise is stable"}
        >>> rng = np.random.RandomState(42)
        >>> data_random = rng.normal(0, 1, 144)
        >>> is_stable(data_random, freq=12, alpha=1.0)
        True

        ```

    ??? tip "See Also"
        - [`stability()`][ts_stat_tests.algorithms.stability.stability]
    """
    return True if _stability(data=data, freq=freq) < alpha else False


# ------------------------------------------------------------------------------#
# Lumpiness                                                                  ####
# ------------------------------------------------------------------------------#


@typechecked
def is_lumpy(
    data: Union[NDArray[np.float64], pd.DataFrame, pd.Series],
    freq: int = 1,
    alpha: float = 0.5,
) -> bool:
    r"""
    !!! note "Summary"
        Determine if a time series is lumpy based on a variance threshold.

    ???+ abstract "Details"
        A time series is considered lumpy if the variance of its windowed variances (the lumpiness metric) exceeds a specified threshold (`alpha`). High lumpiness values indicate that the volatility of the series is inconsistently distributed across time.

    Params:
        data (Union[NDArray[np.float64], pd.DataFrame, pd.Series]):
            The time series data to analyse.
        freq (int):
            The number of observations per seasonal period or the desired window size for tiling.
            Default: `1`
        alpha (float):
            The threshold for lumpiness. The series is considered lumpy if the calculated lumpiness metric is greater than this value.
            Default: `0.5`

    Returns:
        (bool):
            `True` if the lumpiness metric is greater than `alpha` (the series is lumpy), otherwise `False`.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> import numpy as np
        >>> from ts_stat_tests.tests.stability import is_lumpy
        >>> from ts_stat_tests.utils.data import load_airline

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Check if airline data is lumpy"}
        >>> data = load_airline().values
        >>> is_lumpy(data, freq=12, alpha=1.0)
        True

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Check if random noise is lumpy"}
        >>> rng = np.random.RandomState(42)
        >>> data_random = rng.normal(0, 1, 144)
        >>> is_lumpy(data_random, freq=12, alpha=1.0)
        False

        ```

    ??? tip "See Also"
        - [`lumpiness()`][ts_stat_tests.algorithms.stability.lumpiness]
    """
    return True if _lumpiness(data=data, freq=freq) > alpha else False
