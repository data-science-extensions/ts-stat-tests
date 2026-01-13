# ============================================================================ #
#                                                                              #
#     Title: Seasonality                                                       #
#     Purpose: Algorithms for testing seasonality in time series data.         #
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
    Seasonality tests are statistical tests used to determine whether a time series exhibits seasonal patterns or cycles. Seasonality refers to the regular and predictable fluctuations in a time series that occur at specific intervals, such as daily, weekly, monthly, or yearly.

    Seasonality tests help identify whether a time series has a seasonal component that needs to be accounted for in forecasting models. By detecting seasonality, analysts can choose appropriate models that capture these patterns and improve the accuracy of their forecasts.

    Common seasonality tests include the QS test, OCSB test, Canova-Hansen test, and others. These tests analyze the autocorrelation structure of the time series data to identify significant seasonal patterns.

    Overall, seasonality tests are essential tools in time series analysis and forecasting, as they help identify and account for seasonal patterns that can significantly impact the accuracy of predictions.
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
from typing import Optional, Union

# ## Python Third Party Imports ----
import numpy as np
from numpy.typing import ArrayLike
from pmdarima.arima.arima import ARIMA
from pmdarima.arima.auto import auto_arima
from pmdarima.arima.seasonality import CHTest, OCSBTest
from scipy.stats import chi2
from statsmodels.tsa.seasonal import seasonal_decompose  # , STL, DecomposeResult,
from typeguard import typechecked

# ## Local First Party Imports ----
from ts_stat_tests.algorithms.correlation import acf as _acf


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["qs", "ocsb", "ch", "seasonal_strength", "trend_strength", "spikiness"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Algorithms                                                             ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@typechecked
def qs(
    x: ArrayLike,
    freq: int = 0,
    diff: bool = True,
    residuals: bool = False,
    autoarima: bool = True,
) -> Union[tuple[float, float], tuple[float, float, Optional[ARIMA]]]:

    _x: np.ndarray = x if isinstance(x, np.ndarray) else np.asarray(x)
    if np.isnan(_x).all():
        raise AttributeError("All observations are NaN.")
    if diff and residuals:
        print(
            "The differences of the residuals of a non-seasonal ARIMA model are computed and used."
            "It may be better to either only take the differences or use the residuals."
        )
    if freq < 2:
        raise AttributeError(f"The number of observations per cycle is '{freq}', which is too small.")

    model: Optional[ARIMA] = None

    if residuals:
        if autoarima:
            max_order: int = 1 if freq < 8 else 3
            allow_drift: bool = True if freq < 8 else False
            try:
                model = auto_arima(
                    y=_x,
                    max_P=1,
                    max_Q=1,
                    max_p=3,
                    max_q=3,
                    seasonal=False,
                    stepwise=False,
                    max_order=max_order,
                    allow_drift=allow_drift,
                )
            except Exception:  # TODO: specify exception
                try:
                    model = ARIMA(order=(0, 1, 1)).fit(y=_x)
                except Exception:  # TODO: specify exception
                    print("Could not estimate any ARIMA model, original data series is used.")
            if model is not None:
                _x = model.resid()
        else:
            try:
                model = ARIMA(order=(0, 1, 1)).fit(y=_x)
            except Exception:  # TODO: specify exception
                print("Could not estimate any ARIMA model, original data series is used.")
            if model is not None:
                _x = model.resid()

    # Do diff
    y: np.ndarray = np.diff(_x) if diff else _x

    # Pre-check
    if np.nanvar(y[~np.isnan(y)]) == 0:
        raise ValueError(
            "The Series is a constant (possibly after transformations). QS-Test cannot be computed on constants."
        )

    # Test Statistic
    acf_output: np.ndarray = _acf(x=y, nlags=freq * 2, missing="drop")
    rho_output: np.ndarray = acf_output[[freq, freq * 2]]
    rho: np.ndarray = np.array([0, 0]) if np.any(np.array(rho_output) <= 0) else rho_output
    N: int = len(y[~np.isnan(y)])
    QS: float = float(N * (N + 2) * (rho[0] ** 2 / (N - freq) + rho[1] ** 2 / (N - freq * 2)))
    Pval: float = float(chi2.sf(QS, 2))

    if residuals:
        return QS, Pval, model
    return QS, Pval


@typechecked
def ocsb(x: np.ndarray, m: int, lag_method: str = "aic", max_lag: int = 3) -> int:
    return OCSBTest(m=m, lag_method=lag_method, max_lag=max_lag).estimate_seasonal_differencing_term(x)


@typechecked
def ch(x: np.ndarray, m: int) -> int:
    return CHTest(m=m).estimate_seasonal_differencing_term(x)


@typechecked
def seasonal_strength(x: np.ndarray, m: int) -> float:
    decomposition = seasonal_decompose(x=x, period=m, model="additive")
    seasonal = np.nanvar(decomposition.seasonal)
    residual = np.nanvar(decomposition.resid)
    return float(seasonal / (seasonal + residual))


@typechecked
def trend_strength(x: np.ndarray, m: int) -> float:
    decomposition = seasonal_decompose(x=x, period=m, model="additive")
    trend = np.nanvar(decomposition.trend)
    residual = np.nanvar(decomposition.resid)
    return float(trend / (trend + residual))


@typechecked
def spikiness(x: np.ndarray, m: int) -> float:
    decomposition = seasonal_decompose(x=x, model="additive", period=m)
    madr = np.mean(np.abs(decomposition.resid))
    mads = np.mean(np.abs(decomposition.seasonal))
    return float(madr / mads)
