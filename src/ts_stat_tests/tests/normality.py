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
) -> Any:
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
            Defaults to `"dp"`.
        axis (int):
            Axis along which to compute the test. Default is `0`.
        nan_policy (VALID_DP_NAN_POLICY_OPTIONS):
            Defines how to handle when input contains `NaN`.<br>
            - `propagate`: returns `NaN`<br>
            - `raise`: throws an error<br>
            - `omit`: performs the calculations ignoring `NaN` values<br>
            Defaults to `"propagate"`.
        dist (VALID_AD_DIST_OPTIONS):
            The type of distribution to test against.<br>
            Only relevant when `algorithm=anderson`.<br>
            Defaults to `"norm"`.

    Raises:
        ValueError: When the given value for `algorithm` is not valid.

    Returns:
        (Any):
            The result of the normality test.

    !!! Success "Credit"
        Calculations are performed by `scipy.stats` and `statsmodels.stats`.

    ???+ example "Examples"

        `normality` with `dagostino-pearson` algorithm:
        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> import numpy as np
        >>> from ts_stat_tests.tests.normality import normality
        >>> data = np.random.normal(0, 1, 100)
        >>> result = normality(data, algorithm="dp")
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
        return _jb(x=x, axis=axis)
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
) -> dict[str, Union[str, float, bool, Any]]:
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
            Defaults to `"dp"`.
        alpha (float):
            Significance level. Default is `0.05`.
        axis (int):
            Axis along which to compute the test. Default is `0`.
        nan_policy (VALID_DP_NAN_POLICY_OPTIONS):
            Defines how to handle when input contains `NaN`.<br>
            - `propagate`: returns `NaN`<br>
            - `raise`: throws an error<br>
            - `omit`: performs the calculations ignoring `NaN` values<br>
            Defaults to `"propagate"`.
        dist (VALID_AD_DIST_OPTIONS):
            The type of distribution to test against.<br>
            Only relevant when `algorithm=anderson`.<br>
            Defaults to `"norm"`.

    Returns:
        (dict):
            A dictionary containing the results of the test.

    !!! Success "Credit"
        Calculations are performed by `scipy.stats` and `statsmodels.stats`.

    ???+ example "Examples"

        `is_normal` with `dagostino-pearson` algorithm:
        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> import numpy as np
        >>> from ts_stat_tests.tests.normality import is_normal
        >>> data = np.random.normal(0, 1, 100)
        >>> result = is_normal(data, algorithm="dp")
        >>> result["result"]
        True
        ```
    """
    res = normality(x=x, algorithm=algorithm, axis=axis, nan_policy=nan_policy, dist=dist)

    # Anderson-Darling is a bit different
    options: dict[str, tuple[str, ...]] = {
        "ad": ("ad", "anderson", "anderson-darling"),
    }

    if algorithm in options["ad"]:
        # res is AndersonResult(statistic, critical_values, significance_level, fit_result)
        # indexing only gives the first 3 elements
        stat, crit, sig = res[0], res[1], res[2]
        # sig is something like [15. , 10. ,  5. ,  2.5,  1. ]
        # alpha is something like 0.05 (which is 5%)
        idx = np.argmin(np.abs(sig - (alpha * 100)))
        critical_value = crit[idx]
        is_norm = stat < critical_value
        return {
            "result": bool(is_norm),
            "statistic": float(stat),
            "critical_value": float(critical_value),
            "significance_level": float(sig[idx]),
            "alpha": float(alpha),
        }

    # For others, they return (statistic, pvalue) or similar
    if hasattr(res, "pvalue"):
        p_val = res.pvalue
        stat = res.statistic
    elif isinstance(res, (tuple, list)):
        stat, p_val = res[0], res[1]
    else:
        # Fallback
        stat = res
        p_val = None

    is_norm = p_val >= alpha if p_val is not None else False

    return {
        "result": bool(is_norm),
        "statistic": float(stat) if isinstance(stat, (float, int)) else stat,
        "p_value": float(p_val) if p_val is not None else None,
        "alpha": float(alpha),
    }
