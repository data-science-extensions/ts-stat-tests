# ============================================================================ #
#                                                                              #
#     Title   : Correlation Tests                                              #
#     Purpose : This module is a single point of entry for all correlation     #
#         tests in the ts_stat_tests package.                                  #
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
    This module contains tests for the correlation functions defined in the `ts_stat_tests.algorithms.correlation` module.
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
from typing import Literal, Union, overload

# ## Python Third Party Imports ----
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from statsmodels.regression.linear_model import (
    RegressionResults,
    RegressionResultsWrapper,
)
from statsmodels.stats.diagnostic import ResultsStore
from statsmodels.tsa.stattools import ArrayLike1D
from typeguard import typechecked

# ## Local First Party Imports ----
from ts_stat_tests.algorithms.correlation import (
    acf as _acf,
    bglm as _bglm,
    ccf as _ccf,
    lb as _lb,
    lm as _lm,
    pacf as _pacf,
)
from ts_stat_tests.utils.errors import generate_error_message


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["correlation", "is_correlated"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Tests                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@overload
def correlation(
    x: ArrayLike,
    algorithm: Literal["acf", "auto", "ac"],
    **kwargs: Union[float, int, str, bool, ArrayLike, None],
) -> Union[np.ndarray, tuple[np.ndarray, ...]]: ...
@overload
def correlation(
    x: ArrayLike1D,
    algorithm: Literal["pacf", "partial", "pc"],
    **kwargs: Union[float, int, str, bool, ArrayLike, None],
) -> Union[np.ndarray, tuple[np.ndarray, ...]]: ...
@overload
def correlation(
    x: ArrayLike,
    algorithm: Literal["ccf", "cross", "cross-correlation", "cc"],
    **kwargs: Union[float, int, str, bool, ArrayLike, None],
) -> Union[np.ndarray, tuple[np.ndarray, ...]]: ...
@overload
def correlation(
    x: ArrayLike,
    algorithm: Literal["lb", "alb", "acorr_ljungbox", "acor_lb", "a_lb", "ljungbox"],
    **kwargs: Union[float, int, str, bool, ArrayLike, None],
) -> pd.DataFrame: ...
@overload
def correlation(
    x: ArrayLike,
    algorithm: Literal["lm", "alm", "acorr_lm", "a_lm"],
    **kwargs: Union[float, int, str, bool, ArrayLike, None],
) -> Union[
    tuple[float, float, float, float],
    tuple[float, float, float, float, ResultsStore],
]: ...
@overload
def correlation(
    x: Union[RegressionResults, RegressionResultsWrapper],
    algorithm: Literal["bglm", "breusch_godfrey", "bg"],
    **kwargs: Union[float, int, str, bool, ArrayLike, None],
) -> Union[
    tuple[float, float, float, float],
    tuple[float, float, float, float, ResultsStore],
]: ...
@typechecked
def correlation(
    x: Union[ArrayLike, ArrayLike1D, RegressionResults, RegressionResultsWrapper],
    algorithm: str = "acf",
    **kwargs: Union[float, int, str, bool, ArrayLike, None],
) -> Union[
    np.ndarray,
    tuple[np.ndarray, ...],
    pd.DataFrame,
    tuple[float, float, float, float],
    tuple[float, float, float, float, ResultsStore],
]:
    """
    !!! note "Summary"
        A unified interface for various correlation tests.

    ???+ abstract "Details"
        This function acts as a dispatcher for several correlation measures and tests, allowing users to access them through a single, consistent API. Depending on the `algorithm` parameter, it routes the call to the appropriate implementation in `ts_stat_tests.algorithms.correlation`.

        The supported algorithms include:

        - **Autocorrelation Function (ACF)**: Measures the correlation of a signal with a delayed copy of itself.
        - **Partial Autocorrelation Function (PACF)**: Measures the correlation between a signal and its lagged values after removing the effects of intermediate lags.
        - **Cross-Correlation Function (CCF)**: Measures the correlation between two signals at different lags.
        - **Ljung-Box Test**: Tests for the presence of autocorrelation in the residuals of a model.
        - **Lagrange Multiplier (LM) Test**: A generic test for autocorrelation, often used for ARCH effects.
        - **Breusch-Godfrey Test**: A more general version of the LM test for serial correlation in residuals.

    Params:
        x (Union[ArrayLike, ArrayLike1D, RegressionResults, RegressionResultsWrapper]):
            The input time series data or regression results.
        algorithm (str):
            The correlation algorithm to use. Options include:
            - "acf", "auto", "ac": Autocorrelation Function
            - "pacf", "partial", "pc": Partial Autocorrelation Function
            - "ccf", "cross", "cross-correlation", "cc": Cross-Correlation Function
            - "lb", "alb", "acorr_ljungbox", "acor_lb", "a_lb", "ljungbox": Ljung-Box Test
            - "lm", "alm", "acorr_lm", "a_lm": Lagrange Multiplier Test
            - "bglm", "breusch_godfrey", "bg": Breusch-Godfrey Test
        kwargs (Union[float, int, str, bool, ArrayLike, None]):
            Additional keyword arguments specific to the chosen algorithm.

    Returns:
        (Union[np.ndarray, tuple[np.ndarray, ...], pd.DataFrame, tuple[float, float, float, float], tuple[float, float, float, float, ResultsStore]]):
            Returns the result of the specified correlation test.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.tests.correlation import correlation
        >>> from ts_stat_tests.utils.data import data_normal
        >>> normal = data_normal

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Autocorrelation (ACF)"}
        >>> res = correlation(normal, algorithm="acf", nlags=10)
        >>> print(f"Lag 1 ACF: {res[1]:.4f}")
        Lag 1 ACF: 0.0236

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Ljung-Box test"}
        >>> res = correlation(normal, algorithm="lb", lags=[5])
        >>> print(res)
            lb_stat  lb_pvalue
        5  7.882362   0.162839

        ```

    ??? tip "See Also"
        - [`ts_stat_tests.algorithms.correlation.acf`][ts_stat_tests.algorithms.correlation.acf]: Autocorrelation Function algorithm.
        - [`ts_stat_tests.algorithms.correlation.pacf`][ts_stat_tests.algorithms.correlation.pacf]: Partial Autocorrelation Function algorithm.
        - [`ts_stat_tests.algorithms.correlation.ccf`][ts_stat_tests.algorithms.correlation.ccf]: Cross-Correlation Function algorithm.
        - [`ts_stat_tests.algorithms.correlation.lb`][ts_stat_tests.algorithms.correlation.lb]: Ljung-Box Test algorithm.
        - [`ts_stat_tests.algorithms.correlation.lm`][ts_stat_tests.algorithms.correlation.lm]: Lagrange Multiplier Test algorithm.
        - [`ts_stat_tests.algorithms.correlation.bglm`][ts_stat_tests.algorithms.correlation.bglm]: Breusch-Godfrey Test algorithm.
    """

    options: dict[str, tuple[str, ...]] = {
        "acf": ("acf", "auto", "ac"),
        "pacf": ("pacf", "partial", "pc"),
        "ccf": ("ccf", "cross", "cross-correlation", "cc"),
        "lb": ("alb", "acorr_ljungbox", "acor_lb", "a_lb", "lb", "ljungbox"),
        "lm": ("alm", "acorr_lm", "a_lm", "lm"),
        "bglm": ("bglm", "breusch_godfrey", "bg"),
    }

    if algorithm in options["acf"]:
        return _acf(x=x, **kwargs)  # type: ignore

    if algorithm in options["pacf"]:
        return _pacf(x=x, **kwargs)  # type: ignore

    if algorithm in options["lb"]:
        return _lb(x=x, **kwargs)  # type: ignore

    if algorithm in options["lm"]:
        return _lm(resid=x, **kwargs)  # type: ignore

    if algorithm in options["ccf"]:
        if "y" not in kwargs or kwargs["y"] is None:
            raise ValueError("The 'ccf' algorithm requires a 'y' parameter.")
        return _ccf(x=x, **kwargs)  # type: ignore

    if algorithm in options["bglm"]:
        return _bglm(res=x, **kwargs)  # type: ignore

    raise ValueError(
        generate_error_message(
            parameter_name="algorithm",
            value_parsed=algorithm,
            options=options,
        )
    )


@typechecked
def is_correlated() -> None:
    """
    !!! note "Summary"
        A placeholder function for checking if a time series is correlated.
    """
    raise NotImplementedError("is_correlated is a placeholder and has not been implemented yet.")
