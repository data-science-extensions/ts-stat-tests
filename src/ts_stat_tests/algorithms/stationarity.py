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
from arch.unitroot import DFGLS as _ers, VarianceRatio as _vr
from numpy.typing import ArrayLike
from pmdarima.arima import PPTest
from statsmodels.stats.diagnostic import ResultsStore
from statsmodels.tsa.stattools import (
    adfuller as _adfuller,
    kpss as _kpss,
    range_unit_root_test as _rur,
    zivot_andrews as _za,
)
from typeguard import typechecked


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["adf", "kpss", "rur", "za", "pp", "ers", "vr"]


## --------------------------------------------------------------------------- #
##  Constants                                                               ####
## --------------------------------------------------------------------------- #


VALID_ADF_REGRESSION_OPTIONS = Literal["c", "ct", "ctt", "n"]
VALID_ADF_AUTOLAG_OPTIONS = Literal["AIC", "BIC", "t-stat"]
VALID_KPSS_REGRESSION_OPTIONS = Literal["c", "ct"]
VALID_KPSS_NLAGS_OPTIONS = Literal["auto", "legacy"]
VALID_ZA_REGRESSION_OPTIONS = Literal["c", "t", "ct"]
VALID_ZA_AUTOLAG_OPTIONS = Literal["AIC", "BIC", "t-stat"]
VALID_ERS_TREND_OPTIONS = Literal["c", "ct"]
VALID_ERS_METHOD_OPTIONS = Literal["aic", "bic", "t-stat"]
VALID_VR_TREND_OPTIONS = Literal["c", "n"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Algorithms                                                             ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@overload
def adf(
    x: ArrayLike,
    maxlag: Optional[int] = None,
    regression: VALID_ADF_REGRESSION_OPTIONS = "c",
    *,
    autolag: Optional[VALID_ADF_AUTOLAG_OPTIONS] = "AIC",
    store: Literal[True],
    regresults: bool = False,
) -> tuple[float, float, dict, ResultsStore]: ...
@overload
def adf(
    x: ArrayLike,
    maxlag: Optional[int] = None,
    regression: VALID_ADF_REGRESSION_OPTIONS = "c",
    *,
    autolag: None = None,
    store: Literal[False] = False,
    regresults: bool = False,
) -> tuple[float, float, int, int, dict]: ...
@overload
def adf(
    x: ArrayLike,
    maxlag: Optional[int] = None,
    regression: VALID_ADF_REGRESSION_OPTIONS = "c",
    *,
    autolag: VALID_ADF_AUTOLAG_OPTIONS = "AIC",
    store: Literal[False] = False,
    regresults: bool = False,
) -> tuple[float, float, int, int, dict, float]: ...
@typechecked
def adf(
    x: ArrayLike,
    maxlag: Optional[int] = None,
    regression: VALID_ADF_REGRESSION_OPTIONS = "c",
    *,
    autolag: Optional[VALID_ADF_AUTOLAG_OPTIONS] = "AIC",
    store: bool = False,
    regresults: bool = False,
) -> Union[
    tuple[float, float, dict, ResultsStore],
    tuple[float, float, int, int, dict],
    tuple[float, float, int, int, dict, float],
]:
    return _adfuller(x=x, maxlag=maxlag, regression=regression, autolag=autolag, store=store, regresults=regresults)


@overload
def kpss(
    x: ArrayLike,
    regression: VALID_KPSS_REGRESSION_OPTIONS = "c",
    nlags: Optional[Union[VALID_KPSS_NLAGS_OPTIONS, int]] = None,
    *,
    store: Literal[True],
) -> tuple[float, float, dict, ResultsStore]: ...
@overload
def kpss(
    x: ArrayLike,
    regression: VALID_KPSS_REGRESSION_OPTIONS = "c",
    nlags: Optional[Union[VALID_KPSS_NLAGS_OPTIONS, int]] = None,
    *,
    store: Literal[False] = False,
) -> tuple[float, float, int, dict]: ...
@typechecked
def kpss(
    x: ArrayLike,
    regression: VALID_KPSS_REGRESSION_OPTIONS = "c",
    nlags: Optional[Union[VALID_KPSS_NLAGS_OPTIONS, int]] = None,
    *,
    store: bool = False,
) -> Union[
    tuple[float, float, dict, ResultsStore],
    tuple[float, float, int, dict],
]:
    return _kpss(x=x, regression=regression, nlags=nlags, store=store)


@overload
def rur(x: ArrayLike, *, store: Literal[True]) -> tuple[float, float, dict]: ...
@overload
def rur(x: ArrayLike, *, store: Literal[False] = False) -> tuple[float, float, dict, ResultsStore]: ...
@typechecked
def rur(x: ArrayLike, *, store: bool = False) -> Union[
    tuple[float, float, dict],
    tuple[float, float, dict, ResultsStore],
]:
    return _rur(x=x, store=store)


@typechecked
def za(
    x: ArrayLike,
    trim: float = 0.15,
    maxlag: Optional[int] = None,
    regression: VALID_ZA_REGRESSION_OPTIONS = "c",
    autolag: Optional[VALID_ZA_AUTOLAG_OPTIONS] = "AIC",
) -> tuple[float, float, dict, int, int]:
    return _za(x=x, trim=trim, maxlag=maxlag, regression=regression, autolag=autolag)


@typechecked
def pp(
    x: ArrayLike,
    alpha: float = 0.05,
    lshort: bool = True,
) -> tuple[float, bool]:
    return PPTest(alpha=alpha, lshort=lshort).should_diff(x=x)


@typechecked
def ers(
    y: ArrayLike,
    lags: Optional[int] = None,
    trend: VALID_ERS_TREND_OPTIONS = "c",
    max_lags: Optional[int] = None,
    method: VALID_ERS_METHOD_OPTIONS = "aic",
    low_memory: Optional[bool] = None,
) -> tuple[float, float]:
    ers = _ers(
        y=y,
        lags=lags,
        trend=trend,
        max_lags=max_lags,
        method=method,
        low_memory=low_memory,
    )
    return ers.pvalue, ers.stat


@typechecked
def vr(
    y: ArrayLike,
    lags: int = 2,
    trend: VALID_VR_TREND_OPTIONS = "c",
    debiased: bool = True,
    robust: bool = True,
    overlap: bool = True,
) -> tuple[float, float, float]:
    vr = _vr(
        y=y,
        lags=lags,
        trend=trend,
        debiased=debiased,
        robust=robust,
        overlap=overlap,
    )
    return vr.pvalue, vr.stat, vr.vr
