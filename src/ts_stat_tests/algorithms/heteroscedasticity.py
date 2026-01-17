# ============================================================================ #
#                                                                              #
#     Title: Heteroscedasticity Algorithms                                     #
#     Purpose: Implementation of heteroscedasticity tests.                     #
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
    This module implements various heteroscedasticity tests including:
    - ARCH Test
    - Breusch-Pagan Test
    - Goldfeld-Quandt Test
    - White's Test
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
from typing import Literal, Optional, Union, overload

# ## Python Third Party Imports ----
import numpy as np
from numpy.typing import ArrayLike
from statsmodels.sandbox.stats.diagnostic import ResultsStore
from statsmodels.stats.diagnostic import (
    het_arch,
    het_breuschpagan,
    het_goldfeldquandt,
    het_white,
)
from typeguard import typechecked


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = [
    "arch",
    "bpl",
    "gq",
    "wlm",
]


## --------------------------------------------------------------------------- #
##  Constants                                                               ####
## --------------------------------------------------------------------------- #


VALID_GQ_ALTERNATIVES_OPTIONS = Literal["two-sided", "increasing", "decreasing"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Algorithms                                                             ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@overload
def arch(
    resid: ArrayLike, nlags: Optional[int] = None, ddof: int = 0, *, store: Literal[False] = False
) -> tuple[np.float64, np.float64, np.float64, np.float64]: ...
@overload
def arch(
    resid: ArrayLike, nlags: Optional[int] = None, ddof: int = 0, *, store: Literal[True]
) -> tuple[np.float64, np.float64, np.float64, np.float64, ResultsStore]: ...
@typechecked
def arch(resid: ArrayLike, nlags: Optional[int] = None, ddof: int = 0, *, store: bool = False) -> Union[
    tuple[np.float64, np.float64, np.float64, np.float64],
    tuple[np.float64, np.float64, np.float64, np.float64, ResultsStore],
]:
    return het_arch(resid=resid, nlags=nlags, store=store, ddof=ddof)


@typechecked
def bpl(
    resid: ArrayLike, exog_het: ArrayLike, robust: bool = True
) -> tuple[np.float64, np.float64, np.float64, np.float64]:
    return het_breuschpagan(resid=resid, exog_het=exog_het, robust=robust)


@overload
def gq(
    y: ArrayLike,
    x: ArrayLike,
    idx: Optional[int] = None,
    split: Optional[Union[int, float]] = None,
    drop: Optional[Union[int, float]] = None,
    alternative: VALID_GQ_ALTERNATIVES_OPTIONS = "increasing",
    *,
    store: Literal[False] = False,
) -> tuple[np.float64, np.float64, np.float64, np.float64]: ...
@overload
def gq(
    y: ArrayLike,
    x: ArrayLike,
    idx: Optional[int] = None,
    split: Optional[Union[int, float]] = None,
    drop: Optional[Union[int, float]] = None,
    alternative: VALID_GQ_ALTERNATIVES_OPTIONS = "increasing",
    *,
    store: Literal[True],
) -> tuple[np.float64, np.float64, np.float64, np.float64, ResultsStore]: ...
@typechecked
def gq(
    y: ArrayLike,
    x: ArrayLike,
    idx: Optional[int] = None,
    split: Optional[Union[int, float]] = None,
    drop: Optional[Union[int, float]] = None,
    alternative: VALID_GQ_ALTERNATIVES_OPTIONS = "increasing",
    *,
    store: bool = False,
) -> Union[
    tuple[np.float64, np.float64, np.float64, np.float64],
    tuple[np.float64, np.float64, np.float64, np.float64, ResultsStore],
]:
    return het_goldfeldquandt(y=y, x=x, idx=idx, split=split, drop=drop, alternative=alternative, store=store)


@typechecked
def wlm(resid: ArrayLike, exog_het: ArrayLike) -> tuple[np.float64, np.float64, np.float64, np.float64]:
    return het_white(resid=resid, exog_het=exog_het)
