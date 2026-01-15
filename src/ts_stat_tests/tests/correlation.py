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
from numpy.typing import ArrayLike, NDArray
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
) -> Union[NDArray[np.float64], tuple[NDArray[np.float64], ...]]: ...
@overload
def correlation(
    x: ArrayLike1D,
    algorithm: Literal["pacf", "partial", "pc"],
    **kwargs: Union[float, int, str, bool, ArrayLike, None],
) -> Union[NDArray[np.float64], tuple[NDArray[np.float64], ...]]: ...
@overload
def correlation(
    x: ArrayLike,
    algorithm: Literal["ccf", "cross", "cross-correlation", "cc"],
    **kwargs: Union[float, int, str, bool, ArrayLike, None],
) -> Union[NDArray[np.float64], tuple[NDArray[np.float64], ...]]: ...
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
    NDArray[np.float64],
    tuple[NDArray[np.float64], ...],
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

    Raises:
        (ValueError):
            If an unsupported algorithm is specified.

    Returns:
        (Union[NDArray[np.float64], tuple[NDArray[np.float64], ...], pd.DataFrame, tuple[float, float, float, float], tuple[float, float, float, float, ResultsStore]]):
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
def is_correlated(
    x: Union[ArrayLike, ArrayLike1D, RegressionResults, RegressionResultsWrapper],
    algorithm: str = "lb",
    alpha: float = 0.05,
    **kwargs: Union[float, int, str, bool, ArrayLike, None],
) -> dict[str, Union[str, float, bool, None]]:
    """
    !!! note "Summary"
        Test whether a given data set is `correlated` or not.

    ???+ abstract "Details"
        This function checks for autocorrelation in the given data using various tests. By default, it uses the Ljung-Box test.

        - **Ljung-Box (`lb`)**: Tests the null hypothesis that the data are independently distributed (i.e. no autocorrelation). If the p-value is less than `alpha`, the null hypothesis is rejected, and the series is considered `correlated`. If multiple lags are provided, it checks if any of the p-values are below `alpha`.
        - **LM Test (`lm`)**: Tests for serial correlation. If the LMP-value is less than `alpha`, it is considered `correlated`.
        - **Breusch-Godfrey (`bglm`)**: Tests for serial correlation in residuals. If the LMP-value is less than `alpha`, it is considered `correlated`.

    Params:
        x (Union[ArrayLike, ArrayLike1D, RegressionResults, RegressionResultsWrapper]):
            The input time series data or regression results.
        algorithm (str):
            The correlation algorithm to use. Options include:
            - `"lb"`, `"alb"`, `"acorr_ljungbox"`, `"acor_lb"`, `"a_lb"`, `"ljungbox"`: Ljung-Box Test (default)
            - `"lm"`, `"alm"`, `"acorr_lm"`, `"a_lm"`: Lagrange Multiplier Test
            - `"bglm"`, `"breusch_godfrey"`, `"bg"`: Breusch-Godfrey Test
        alpha (float, optional):
            The significance level for the test. Default: `0.05`.
        kwargs (Union[float, int, str, bool, ArrayLike, None]):
            Additional arguments to pass to the underlying algorithm.

    Raises:
        (ValueError):
            If an unsupported algorithm is specified.

    Returns:
        (dict[str, Union[str, float, bool, None]]):
            A dictionary containing:
            - `"result"` (bool): `True` if the series is significantly correlated.
            - `"statistic"` (float): The test statistic.
            - `"pvalue"` (float): The p-value of the test.
            - `"alpha"` (float): The significance level used.
            - `"algorithm"` (str): The algorithm name used.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.tests.correlation import is_correlated
        >>> from ts_stat_tests.utils.data import data_normal
        >>> normal = data_normal

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Ljung-Box test on random data"}
        >>> res = is_correlated(normal, algorithm="lb", lags=[5])
        >>> res["result"]
        False
        >>> print(f"p-value: {res['pvalue']:.4f}")
        p-value: 0.1628

        ```

        ```pycon {.py .python linenums="1" title="Example 2: LM test"}
        >>> res = is_correlated(normal, algorithm="lm", nlags=5)
        >>> res["result"]
        False

        ```

    ??? tip "See Also"
        - [`correlation()`][ts_stat_tests.tests.correlation.correlation]: Dispatcher for correlation measures and tests.
        - [`ts_stat_tests.algorithms.correlation.lb`][ts_stat_tests.algorithms.correlation.lb]: Ljung-Box Test algorithm.
        - [`ts_stat_tests.algorithms.correlation.lm`][ts_stat_tests.algorithms.correlation.lm]: Lagrange Multiplier Test algorithm.
        - [`ts_stat_tests.algorithms.correlation.bglm`][ts_stat_tests.algorithms.correlation.bglm]: Breusch-Godfrey Test algorithm.
    """
    options: dict[str, tuple[str, ...]] = {
        "lb": ("alb", "acorr_ljungbox", "acor_lb", "a_lb", "lb", "ljungbox"),
        "lm": ("alm", "acorr_lm", "a_lm", "lm"),
        "bglm": ("bglm", "breusch_godfrey", "bg"),
    }

    res = correlation(x=x, algorithm=algorithm, **kwargs)  # type: ignore

    is_corr: bool = False
    stat: float = 0.0
    pval: Union[float, None] = None

    if algorithm in options["lb"]:
        df = res
        # Check if any p-value is significant
        pval = float(df["lb_pvalue"].min())
        # Metric: if any lag shows correlation, the series is correlated
        is_corr = bool(pval < alpha)
        # Return the statistic for the most significant lag
        idx = df["lb_pvalue"].idxmin()
        stat = float(df.loc[idx, "lb_stat"])

    elif algorithm in options["lm"] or algorithm in options["bglm"]:
        # returns (lm, lmpval, fval, fpval)
        res_tuple = res
        stat = float(res_tuple[0])
        pval = float(res_tuple[1])
        is_corr = bool(pval < alpha)

    else:
        raise ValueError(
            f"Algorithm '{algorithm}' is not supported for 'is_correlated'. "
            f"Supported algorithms for boolean check are: 'lb', 'lm', 'bglm'."
        )

    return {
        "result": is_corr,
        "statistic": stat,
        "pvalue": pval,
        "alpha": alpha,
        "algorithm": algorithm,
    }
