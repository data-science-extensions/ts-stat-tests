# ============================================================================ #
#                                                                              #
#     Title: Seasonality Tests                                                 #
#     Purpose: Tests for seasonality detection algorithms.                     #
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
    This module contains functions to assess the seasonality of time series data.

    The implemented algorithms include:

    - QS Test
    - OCSB Test
    - CH Test
    - Seasonal Strength
    - Trend Strength
    - Spikiness

    Each function is designed to analyze a univariate time series and return relevant statistics or indicators of seasonality. This module provides both a dispatcher for flexible algorithm selection and a boolean check for easy integration into pipelines.
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
from typing import Any, Callable, Optional, Union

# ## Python Third Party Imports ----
from numpy.typing import ArrayLike
from pmdarima.arima import ARIMA
from typeguard import typechecked

# ## Local First Party Imports ----
from ts_stat_tests.algorithms.seasonality import (
    ch as _ch,
    ocsb as _ocsb,
    qs as _qs,
    seasonal_strength as _seasonal_strength,
    spikiness as _spikiness,
    trend_strength as _trend_strength,
)
from ts_stat_tests.utils.errors import generate_error_message


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["seasonality", "is_seasonal"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Tests                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@typechecked
def seasonality(
    x: ArrayLike,
    algorithm: str = "qs",
    **kwargs: Union[float, int, str, bool, ArrayLike, None],
) -> Union[float, int, tuple[Union[float, int, ARIMA, None], ...]]:
    """
    !!! note "Summary"
        Dispatcher for seasonality algorithms. This function provides a unified interface to call various seasonality tests.

    ???+ abstract "Details"

        The `seasonality` function acts as a centralized dispatcher for the various seasonality algorithms implemented in the `algorithms.seasonality` module. It allows users to easily switch between different tests by specifying the `algorithm` name.

        The supported algorithms include:

        - `"qs"`: The QS (Quenouille-Sarle) test for seasonality.
        - `"ocsb"`: The Osborn-Chui-Smith-Birchenhall test for seasonal differencing.
        - `"ch"`: The Canova-Hansen test for seasonal stability.
        - `"seasonal_strength"` (or `"ss"`): The STL-based seasonal strength measure.
        - `"trend_strength"` (or `"ts"`): The STL-based trend strength measure.
        - `"spikiness"`: The STL-based spikiness measure.

    Params:
        x (ArrayLike):
            The data to be checked.
        algorithm (str, optional):
            Which seasonality algorithm to use.<br>
            Default: `"qs"`
        kwargs (Union[float, int, str, bool, ArrayLike, None]):
            Additional arguments to pass to the underlying algorithm.

    Returns:
        (Union[float, int, tuple[Union[float, int, object, None], ...]]):
            The result of the seasonality test. The return type depends on the chosen algorithm:
            - `"qs"` returns a tuple `(statistic, pvalue)`.
            - `"ocsb"` and `"ch"` return an integer (0 or 1).
            - `"seasonal_strength"`, `"trend_strength"`, and `"spikiness"` return a float.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.tests.seasonality import seasonality
        >>> data = load_airline().values
        >>> # Using the default QS test
        >>> seasonality(x=data, freq=12)
        (194.469289..., 5.909223...-43)
        >>> # Using seasonal strength
        >>> seasonality(x=data, algorithm="ss", m=12)
        0.778721...

        ```
    """
    options: dict[str, tuple[str, ...]] = {
        "qs": ("qs",),
        "ocsb": ("ocsb",),
        "ch": ("ch",),
        "seasonal_strength": ("seasonal_strength", "ss"),
        "trend_strength": ("trend_strength", "ts"),
        "spikiness": ("spikiness",),
    }

    # Internal helper to handle kwargs casting for ty
    def _call(
        func: Callable[..., Any],
        **args: Any,
    ) -> Any:
        """
        !!! note "Summary"
            Internal helper to call the test function.

        Params:
            func (Callable[..., Any]):
                The function to call.
            args (Any):
                The arguments to pass.

        Returns:
            (Any):
                The result.

        ???+ example "Examples"

            ```python
            # Internal helper.
            ```
        """
        return func(**args)

    if algorithm in options["qs"]:
        return _call(_qs, x=x, **kwargs)
    if algorithm in options["ocsb"]:
        return _call(_ocsb, x=x, **kwargs)
    if algorithm in options["ch"]:
        return _call(_ch, x=x, **kwargs)
    if algorithm in options["seasonal_strength"]:
        return _call(_seasonal_strength, x=x, **kwargs)
    if algorithm in options["trend_strength"]:
        return _call(_trend_strength, x=x, **kwargs)
    if algorithm in options["spikiness"]:
        return _call(_spikiness, x=x, **kwargs)

    raise ValueError(
        generate_error_message(
            parameter_name="algorithm",
            value_parsed=algorithm,
            options={k: str(v) for k, v in options.items()},
        )
    )


@typechecked
def is_seasonal(
    x: ArrayLike,
    algorithm: str = "qs",
    alpha: float = 0.05,
    **kwargs: Union[float, int, str, bool, ArrayLike, None],
) -> dict[str, Union[str, float, bool, None]]:
    """
    !!! note "Summary"
        Boolean check for seasonality. This function wraps the `seasonality` dispatcher and returns a standardized dictionary indicating whether the series is seasonal based on a significance level or threshold.

    ???+ abstract "Details"

        The `is_seasonal` function interprets the results of the underlying seasonality tests to provide a boolean `"result"`.

        - For `"qs"`, the test is considered seasonal if the p-value is less than `alpha`.
        - For `"ocsb"` and `"ch"`, the test is considered seasonal if the returned integer is 1.
        - For `"seasonal_strength"`, the test is considered seasonal if the strength is greater than 0.64 (a common threshold in literature).
        - For others, it checks if the statistic is greater than 0.

    Params:
        x (ArrayLike):
            The data to be checked.
        algorithm (str, optional):
            Which seasonality algorithm to use.<br>
            Default: `"qs"`
        alpha (float, optional):
            The significance level for the test (used by `"qs"`).<br>
            Default: `0.05`
        kwargs (Union[float, int, str, bool, ArrayLike, None]):
            Additional arguments to pass to the underlying algorithm.

    Returns:
        (dict[str, Union[str, float, bool, None]]):
            A dictionary containing:

            - `"result"` (bool): Indicator if the series is seasonal.
            - `"statistic"` (float): The test statistic (or strength).
            - `"pvalue"` (float, optional): The p-value of the test (if available).
            - `"alpha"` (float): The significance level used.
            - `"algorithm"` (str): The algorithm used.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Standard check"}
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.tests.seasonality import is_seasonal
        >>> data = load_airline().values
        >>> res = is_seasonal(x=data, algorithm="qs", freq=12)
        >>> res["result"]
        True
        >>> res["algorithm"]
        'qs'

        ```
    """
    res: Any = seasonality(x=x, algorithm=algorithm, **kwargs)

    is_sea: bool = False
    stat: float = 0.0
    pval: Optional[float] = None

    if algorithm in ("qs",):
        if isinstance(res, (tuple, list)):
            v0: Any = res[0]
            v1: Any = res[1]
            stat = float(v0) if isinstance(v0, (int, float)) else 0.0
            pval = float(v1) if isinstance(v1, (int, float)) else 1.0
            is_sea = bool(pval < alpha)
    elif algorithm in ("ocsb", "ch"):
        if isinstance(res, (tuple, list)):
            v0: Any = res[0]
            stat = float(v0) if isinstance(v0, (int, float)) else 0.0
        else:
            v_any: Any = res
            stat = float(v_any) if isinstance(res, (int, float)) else 0.0
        is_sea = bool(stat == 1)
    elif algorithm in ("seasonal_strength", "ss"):
        if isinstance(res, (tuple, list)):
            v0: Any = res[0]
            stat = float(v0) if isinstance(v0, (int, float)) else 0.0
        else:
            v_any: Any = res
            stat = float(v_any) if isinstance(res, (int, float)) else 0.0
        # Default threshold of 0.64 is often used for seasonal strength
        is_sea = bool(stat > 0.64)
    else:
        if isinstance(res, (tuple, list)):
            v0: Any = res[0]
            stat = float(v0) if isinstance(v0, (int, float)) else 0.0
        else:
            v_any: Any = res
            stat = float(v_any) if isinstance(res, (int, float)) else 0.0
        is_sea = bool(stat > 0)

    return {
        "result": is_sea,
        "statistic": stat,
        "pvalue": pval,
        "alpha": alpha,
        "algorithm": algorithm,
    }
