# ============================================================================ #
#                                                                              #
#     Title: Normality Tests                                                   #
#     Purpose: Convenience functions for normality algorithms.                 #
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
    This module contains convenience functions and tests for normality measures, allowing for easy access to different normality algorithms.
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
from typing import Any, Union

# ## Python Third Party Imports ----
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats._morestats import AndersonResult, ShapiroResult
from scipy.stats._stats_py import NormaltestResult
from typeguard import typechecked

# ## Local First Party Imports ----
from ts_stat_tests.algorithms.normality import (
    VALID_AD_DIST_OPTIONS,
    VALID_DP_NAN_POLICY_OPTIONS,
    ad as _ad,
    dp as _dp,
    jb as _jb,
    ob as _ob,
    sw as _sw,
)
from ts_stat_tests.utils.errors import generate_error_message


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["normality", "is_normal"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Tests                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@typechecked
def normality(
    x: ArrayLike,
    algorithm: str = "dp",
    axis: int = 0,
    nan_policy: VALID_DP_NAN_POLICY_OPTIONS = "propagate",
    dist: VALID_AD_DIST_OPTIONS = "norm",
) -> Union[tuple[float, ...], NormaltestResult, ShapiroResult, AndersonResult]:
    """
    !!! note "Summary"
        Perform a normality test on the given data.

    ???+ abstract "Details"
        This function is a convenience wrapper around the five underlying algorithms:<br>
        - [`jb()`][ts_stat_tests.algorithms.normality.jb]<br>
        - [`ob()`][ts_stat_tests.algorithms.normality.ob]<br>
        - [`sw()`][ts_stat_tests.algorithms.normality.sw]<br>
        - [`dp()`][ts_stat_tests.algorithms.normality.dp]<br>
        - [`ad()`][ts_stat_tests.algorithms.normality.ad]

    Params:
        x (ArrayLike):
            The data to be checked. Should be a `1-D` or `N-D` data array.
        algorithm (str):
            Which normality algorithm to use.<br>
            - `jb()`: `["jb", "jarque", "jarque-bera"]`<br>
            - `ob()`: `["ob", "omni", "omnibus"]`<br>
            - `sw()`: `["sw", "shapiro", "shapiro-wilk"]`<br>
            - `dp()`: `["dp", "dagostino", "dagostino-pearson"]`<br>
            - `ad()`: `["ad", "anderson", "anderson-darling"]`<br>
            Default: `"dp"`
        axis (int):
            Axis along which to compute the test.
            Default: `0`
        nan_policy (VALID_DP_NAN_POLICY_OPTIONS):
            Defines how to handle when input contains `NaN`.<br>
            - `propagate`: returns `NaN`<br>
            - `raise`: throws an error<br>
            - `omit`: performs the calculations ignoring `NaN` values<br>
            Default: `"propagate"`
        dist (VALID_AD_DIST_OPTIONS):
            The type of distribution to test against.<br>
            Only relevant when `algorithm=anderson`.<br>
            Default: `"norm"`

    Raises:
        (ValueError):
            When the given value for `algorithm` is not valid.

    Returns:
        (Union[tuple[float, float], tuple[float, list[float], list[float]]]):
            If not `"ad"`, returns a `tuple` of `(stat, pvalue)`.
            If `"ad"`, returns a `tuple` of `(stat, critical_values, significance_level)`.

    !!! success "Credit"
        Calculations are performed by `scipy.stats` and `statsmodels.stats`.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.tests.normality import normality
        >>> from ts_stat_tests.utils.data import data_normal
        >>> normal = data_normal

        ```

        ```pycon {.py .python linenums="1" title="Example 1: D'Agostino-Pearson test"}
        >>> stat, pvalue = normality(normal, algorithm="dp")
        >>> print(f"DP statistic: {stat:.4f}")
        DP statistic: 1.3537
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.5082

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Jarque-Bera test"}
        >>> stat, pvalue = normality(normal, algorithm="jb")
        >>> print(f"JB statistic: {stat:.4f}")
        JB statistic: 1.4168
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.4924

        ```
    """
    options: dict[str, tuple[str, ...]] = {
        "jb": ("jb", "jarque", "jarque-bera"),
        "ob": ("ob", "omni", "omnibus"),
        "sw": ("sw", "shapiro", "shapiro-wilk"),
        "dp": ("dp", "dagostino", "dagostino-pearson"),
        "ad": ("ad", "anderson", "anderson-darling"),
    }
    if algorithm in options["jb"]:
        res_jb = _jb(x=x, axis=axis)
        return (res_jb[0], res_jb[1])
    if algorithm in options["ob"]:
        return _ob(x=x, axis=axis)
    if algorithm in options["sw"]:
        return _sw(x=x)
    if algorithm in options["dp"]:
        return _dp(x=x, axis=axis, nan_policy=nan_policy)
    if algorithm in options["ad"]:
        return _ad(x=x, dist=dist)

    raise ValueError(
        generate_error_message(
            parameter_name="algorithm",
            value_parsed=algorithm,
            options=options,
        )
    )


@typechecked
def is_normal(
    x: ArrayLike,
    algorithm: str = "dp",
    alpha: float = 0.05,
    axis: int = 0,
    nan_policy: VALID_DP_NAN_POLICY_OPTIONS = "propagate",
    dist: VALID_AD_DIST_OPTIONS = "norm",
) -> dict[str, Union[str, float, bool, None]]:
    """
    !!! note "Summary"
        Test whether a given data set is `normal` or not.

    ???+ abstract "Details"
        This function implements the given algorithm (defined in the parameter `algorithm`), and returns a dictionary containing the relevant data:
        ```python
        {
            "result": ...,  # The result of the test. Will be `True` if `p-value >= alpha`, and `False` otherwise
            "statistic": ...,  # The test statistic
            "p_value": ...,  # The p-value of the test (if applicable)
            "alpha": ...,  # The significance level used
        }
        ```

    Params:
        x (ArrayLike):
            The data to be checked. Should be a `1-D` or `N-D` data array.
        algorithm (str):
            Which normality algorithm to use.<br>
            - `jb()`: `["jb", "jarque", "jarque-bera"]`<br>
            - `ob()`: `["ob", "omni", "omnibus"]`<br>
            - `sw()`: `["sw", "shapiro", "shapiro-wilk"]`<br>
            - `dp()`: `["dp", "dagostino", "dagostino-pearson"]`<br>
            - `ad()`: `["ad", "anderson", "anderson-darling"]`<br>
            Default: `"dp"`
        alpha (float):
            Significance level.
            Default: `0.05`
        axis (int):
            Axis along which to compute the test.
            Default: `0`
        nan_policy (VALID_DP_NAN_POLICY_OPTIONS):
            Defines how to handle when input contains `NaN`.<br>
            - `propagate`: returns `NaN`<br>
            - `raise`: throws an error<br>
            - `omit`: performs the calculations ignoring `NaN` values<br>
            Default: `"propagate"`
        dist (VALID_AD_DIST_OPTIONS):
            The type of distribution to test against.<br>
            Only relevant when `algorithm=anderson`.<br>
            Default: `"norm"`

    Returns:
        (dict[str, Union[str, float, bool, None]]):
            A dictionary containing:
            - `"result"` (bool): Indicator if the series is normal.
            - `"statistic"` (float): The test statistic.
            - `"p_value"` (float): The p-value of the test (if applicable).
            - `"alpha"` (float): The significance level used.

    !!! success "Credit"
        Calculations are performed by `scipy.stats` and `statsmodels.stats`.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.tests.normality import is_normal
        >>> from ts_stat_tests.utils.data import data_normal, data_random
        >>> normal = data_normal
        >>> random = data_random

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Test normal data"}
        >>> res = is_normal(normal, algorithm="dp")
        >>> res["result"]
        True
        >>> print(f"p-value: {res['p_value']:.4f}")
        p-value: 0.5082

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Test non-normal (random) data"}
        >>> res = is_normal(random, algorithm="sw")
        >>> res["result"]
        False

        ```
    """
    res: Any = normality(x=x, algorithm=algorithm, axis=axis, nan_policy=nan_policy, dist=dist)

    if algorithm in ("ad", "anderson", "anderson-darling"):
        # res is AndersonResult(statistic, critical_values, significance_level, fit_result)
        # indexing only gives the first 3 elements
        res_list: list[Any] = list(res) if isinstance(res, (tuple, list)) else []
        if len(res_list) >= 3:
            v0: Any = res_list[0]
            v1: Any = res_list[1]
            v2: Any = res_list[2]
            stat = v0
            crit = v1
            sig = v2

            # sig is something like [15. , 10. ,  5. ,  2.5,  1. ]
            # alpha is something like 0.05 (which is 5%)
            sig_arr = np.asarray(sig)
            crit_arr = np.asarray(crit)
            idx = np.argmin(np.abs(sig_arr - (alpha * 100)))
            critical_value = crit_arr[idx]
            is_norm = stat < critical_value
            return {
                "result": bool(is_norm),
                "statistic": float(stat),
                "critical_value": float(critical_value),
                "significance_level": float(sig_arr[idx]),
                "alpha": float(alpha),
            }
        # Fallback for unexpected return format
        return {
            "result": False,
            "statistic": 0.0,
            "alpha": float(alpha),
        }

    # For others, they return (statistic, pvalue) or similar
    p_val: Union[float, None] = None
    stat_val: Union[float, None] = None

    # Use getattr to avoid type checker attribute issues
    p_val_attr = getattr(res, "pvalue", None)
    stat_val_attr = getattr(res, "statistic", None)

    if p_val_attr is not None and stat_val_attr is not None:
        p_val = float(p_val_attr)
        stat_val = float(stat_val_attr)
    elif isinstance(res, (tuple, list)) and len(res) >= 2:
        res_tuple: Any = res
        stat_val = float(res_tuple[0])
        p_val = float(res_tuple[1])
    else:
        # Fallback
        if isinstance(res, (float, int)):
            stat_val = float(res)
        p_val = None

    is_norm_val = p_val >= alpha if p_val is not None else False

    return {
        "result": bool(is_norm_val),
        "statistic": stat_val,
        "p_value": p_val,
        "alpha": float(alpha),
    }
