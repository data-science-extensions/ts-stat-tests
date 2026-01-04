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
from typing import Any, Literal, Union, overload

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
    **kwargs: Any,
) -> Union[np.ndarray, tuple[np.ndarray, ...]]: ...
@overload
def correlation(
    x: ArrayLike1D,
    algorithm: Literal["pacf", "partial", "pc"],
    **kwargs: Any,
) -> Union[np.ndarray, tuple[np.ndarray, ...]]: ...
@overload
def correlation(
    x: ArrayLike,
    algorithm: Literal["ccf", "cross", "cross-correlation", "cc"],
    **kwargs: Any,
) -> Union[np.ndarray, tuple[np.ndarray, ...]]: ...
@overload
def correlation(
    x: ArrayLike,
    algorithm: Literal["lb", "alb", "acorr_ljungbox", "acor_lb", "a_lb", "ljungbox"],
    **kwargs: Any,
) -> pd.DataFrame: ...
@overload
def correlation(
    x: ArrayLike,
    algorithm: Literal["lm", "alm", "acorr_lm", "a_lm"],
    **kwargs: Any,
) -> Union[
    tuple[float, float, float, float],
    tuple[float, float, float, float, ResultsStore],
]: ...
@overload
def correlation(
    x: Union[RegressionResults, RegressionResultsWrapper],
    algorithm: Literal["bglm", "breusch_godfrey", "bg"],
    **kwargs: Any,
) -> Union[
    tuple[float, float, float, float],
    tuple[float, float, float, float, ResultsStore],
]: ...
@typechecked
def correlation(
    x: Union[ArrayLike, ArrayLike1D, RegressionResults, RegressionResultsWrapper],
    algorithm: str = "acf",
    **kwargs: Any,
) -> Any:
    """
    !!! note "Summary"
        A unified interface for various correlation tests.

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
        kwargs (Any):
            Additional keyword arguments specific to the chosen algorithm.

    Returns:
        (Any):
            The result of the specified correlation test.
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
