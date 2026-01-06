# ---------------------------------------------------------------------------- #
#                                                                              #
#    Setup                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Imports                                                                   ####
# ---------------------------------------------------------------------------- #


# ## Python StdLib Imports ----
from typing import Literal, Optional, Union

# ## Python Third Party Imports ----
from numpy.typing import ArrayLike
from scipy.stats import anderson as _ad, normaltest as _dp, shapiro as _sw
from scipy.stats._morestats import ShapiroResult
from scipy.stats._result_classes import FitResult
from scipy.stats._stats_py import NormaltestResult
from statsmodels.stats.stattools import jarque_bera as _jb, omni_normtest as _ob


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["jb", "ob", "sw", "dp", "ad"]


## --------------------------------------------------------------------------- #
##  Constants                                                               ####
## --------------------------------------------------------------------------- #


VALID_DP_NAN_POLICY_OPTIONS = Literal["propagate", "raise", "omit"]


VALID_AD_DIST_OPTIONS = Literal[
    "norm", "expon", "logistic", "gumbel", "gumbel_l", "gumbel_r", "extreme1", "weibull_min"
]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Algorithms                                                             ####
#                                                                              #
# ---------------------------------------------------------------------------- #


def jb(
    x: ArrayLike,
    axis: int = 0,
) -> tuple[
    Union[float, ArrayLike],
    Union[float, ArrayLike],
    Union[float, ArrayLike],
    Union[float, ArrayLike],
]:
    return _jb(resids=x, axis=axis)


def ob(
    x: ArrayLike,
    axis: int = 0,
) -> Union[tuple[float, tuple], NormaltestResult]:
    return _ob(resids=x, axis=axis)


def sw(
    x: ArrayLike,
) -> Union[tuple[float, float], ShapiroResult]:
    return _sw(x=x)


def dp(
    x: ArrayLike,
    axis: int = 0,
    nan_policy: VALID_DP_NAN_POLICY_OPTIONS = "propagate",
) -> Union[
    tuple[Union[float, ArrayLike], Union[float, ArrayLike]],
    NormaltestResult,
]:
    return _dp(a=x, axis=axis, nan_policy=nan_policy)


def ad(
    x: ArrayLike,
    dist: Optional[VALID_AD_DIST_OPTIONS] = "norm",
) -> tuple[float, list, list, FitResult]:
    return _ad(x=x, dist=dist)
