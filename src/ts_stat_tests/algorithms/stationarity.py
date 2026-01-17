# ============================================================================ #
#                                                                              #
#     Title: Stationarity Algorithms                                           #
#     Purpose: Algorithms to test for stationarity in time series data.        #
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
    Stationarity tests are statistical tests used to determine whether a time series is stationary or not. A stationary time series is one whose statistical properties, such as mean and variance, do not change over time. Stationarity is an important assumption in many time series forecasting models, as it allows for the use of techniques such as autoregression and moving averages.

    There are several different types of stationarity tests, including the Augmented Dickey-Fuller (ADF) test, the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test, the Phillips-Perron (PP) test, the Elliott-Rothenberg-Stock (ERS) test, and the Variance Ratio (VR) test. Each of these tests has its own strengths and weaknesses, and the choice of which test to use will depend on the specific characteristics of the time series being analyzed.

    Overall, stationarity tests are an important tool in time series analysis and forecasting, as they help identify whether a time series is stationary or non-stationary, which can have implications for the choice of forecasting models and methods.

    For a really good article on ADF & KPSS tests, check: [When A Time Series Only Quacks Like A Duck: Testing for Stationarity Before Running Forecast Models. With Python. And A Duckling Picture.](https://towardsdatascience.com/when-a-time-series-only-quacks-like-a-duck-10de9e165e)
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
from typing import Any, Literal, Optional, Union, overload

# ## Python Third Party Imports ----
import numpy as np
from arch.unitroot import (
    DFGLS as _ers,
    PhillipsPerron as _pp,
    VarianceRatio as _vr,
)
from numpy.typing import ArrayLike
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
VALID_PP_TREND_OPTIONS = Literal["n", "c", "ct"]
VALID_PP_TEST_TYPE_OPTIONS = Literal["rho", "tau"]
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
    autolag: None,
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
    r"""
    !!! note "Summary"
        The Augmented Dickey-Fuller test can be used to test for a unit root in a univariate process in the presence of serial correlation.

    ???+ abstract "Details"

        The Augmented Dickey-Fuller (ADF) test is a statistical test used to determine whether a time series is stationary or not. Stationarity refers to the property of a time series where the statistical properties, such as mean and variance, remain constant over time. Stationarity is important for time series forecasting as it allows for the use of many popular forecasting models, such as ARIMA.

        The ADF test is an extension of the Dickey-Fuller test and involves regressing the first-difference of the time series on its lagged values, and then testing whether the coefficient of the lagged first-difference term is statistically significant. If it is, then the time series is considered non-stationary.

        The null hypothesis of the ADF test is that the time series has a unit root, which means that it is non-stationary. The alternative hypothesis is that the time series is stationary. If the p-value of the test is less than a chosen significance level, typically 0.05, then we reject the null hypothesis and conclude that the time series is stationary.

        In practical terms, if a time series is found to be non-stationary by the ADF test, one can apply differencing to the time series until it becomes stationary. This involves taking the difference between consecutive observations and potentially repeating this process until the time series is stationary.

    Params:
        x (ArrayLike):
            The data series to test.
        maxlag (Optional[int]):
            Maximum lag which is included in test, default value of $12 \times (\frac{nobs}{100})^{\frac{1}{4}}$ is used when `None`.
            Default: `None`
        regression (VALID_ADF_REGRESSION_OPTIONS):
            Constant and trend order to include in regression.

            - `"c"`: constant only (default).
            - `"ct"`: constant and trend.
            - `"ctt"`: constant, and linear and quadratic trend.
            - `"n"`: no constant, no trend.

            Default: `"c"`
        autolag (Optional[VALID_ADF_AUTOLAG_OPTIONS]):
            Method to use when automatically determining the lag length among the values $0, 1, ..., maxlag$.

            - If `"AIC"` (default) or `"BIC"`, then the number of lags is chosen to minimize the corresponding information criterion.
            - `"t-stat"` based choice of `maxlag`. Starts with `maxlag` and drops a lag until the t-statistic on the last lag length is significant using a 5%-sized test.
            - If `None`, then the number of included lags is set to `maxlag`.

            Default: `"AIC"`
        store (bool):
            If `True`, then a result instance is returned additionally to the `adf` statistic.
            Default: `False`
        regresults (bool):
            If `True`, the full regression results are returned.
            Default: `False`

    Returns:
        (Union[tuple[float, float, dict, ResultsStore], tuple[float, float, int, int, dict], tuple[float, float, int, int, dict, float]]):
            Depending on parameters, returns a tuple containing:
            - `adf` (float): The test statistic.
            - `pvalue` (float): MacKinnon's approximate p-value.
            - `uselag` (int): The number of lags used.
            - `nobs` (int): The number of observations used.
            - `critical_values` (dict): Critical values at the 1%, 5%, and 10% levels.
            - `icbest` (float): The maximized information criterion (if `autolag` is not `None`).
            - `resstore` (Optional[ResultsStore]): Result instance (if `store` is `True`).

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.stationarity import adf
        >>> from ts_stat_tests.utils.data import data_airline, data_normal
        >>> normal = data_normal
        >>> airline = data_airline.values

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Stationary Series"}
        >>> stat, pvalue, lags, nobs, crit, icbest = adf(x=normal)
        >>> print(f"ADF statistic: {stat:.4f}")
        ADF statistic: -30.7838
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.0000

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Airline Passengers Data"}
        >>> stat, pvalue, lags, nobs, crit, icbest = adf(x=airline)
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.9919

        ```

        ```pycon {.py .python linenums="1" title="Example 3: Store Result Instance"}
        >>> res = adf(x=airline, store=True)
        >>> print(res)
        (0.8153..., 0.9918..., {'1%': np.float64(-3.4816...), '5%': np.float64(-2.8840...), '10%': np.float64(-2.5787...)}, <statsmodels.stats.diagnostic.ResultsStore object at ...>)

        ```

        ```pycon {.py .python linenums="1" title="Example 4: No Autolag"}
        >>> stat, pvalue, lags, nobs, crit = adf(x=airline, autolag=None, maxlag=5)
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.7670

        ```

    ??? equation "Calculation"

        The mathematical equation for the Augmented Dickey-Fuller (ADF) test for stationarity in time series forecasting is:

        $$
        \Delta y_t = \alpha + \beta y_{t-1} + \sum_{i=1}^p \delta_i \Delta y_{t-i} + \epsilon_t
        $$

        where:

        - $y_t$ is the value of the time series at time $t$.
        - $\Delta y_t$ is the first difference of $y_t$, which is defined as $\Delta y_t = y_t - y_{t-1}$.
        - $\alpha$ is the constant term.
        - $\beta$ is the coefficient on $y_{t-1}$.
        - $\delta_i$ are the coefficients on the lagged differences of $y$.
        - $\epsilon_t$ is the error term.

        The ADF test involves testing the null hypothesis that $\beta = 0$, or equivalently, that the time series has a unit root. If $\beta$ is significantly different from $0$, then the null hypothesis can be rejected and the time series is considered stationary.

        Here are the detailed steps for how to calculate the ADF test:

        1. Collect your time series data and plot it to visually check for any trends, seasonal patterns, or other patterns that could make the data non-stationary. If you detect any such patterns, you will need to pre-process your data (e.g., detrending, deseasonalizing, etc.) to remove these effects.

        1. Calculate the first differences of the time series, which is simply the difference between each observation and the previous observation. This step is performed to transform the original data into a stationary process. The first difference of $y_t$ is defined as $\Delta y_t = y_t - y_{t-1}$.

        1. Estimate the parameters $\alpha$, $\beta$, and $\delta_i$ using the least squares method. This involves regressing $\Delta y_t$ on its lagged values, $y_{t-1}$, and the lagged differences of $y, \Delta y_{t-1}, \Delta y_{t-2}, \dots, \Delta y_{t-p}$, where $p$ is the number of lags to include in the model. The estimated equation is:

            $$
            \Delta y_t = \alpha + \beta y_{t-1} + \sum_{i=1}^p \delta_i \Delta y_{t-i} + \epsilon_t
            $$

        1. Calculate the test statistic, which is given by:

            $$
            ADF = \frac {\beta-1}{SE(\beta)}
            $$

            - where $SE(\beta)$ is the standard error of the coefficient on $y_{t-1}$.

            The test statistic measures the number of standard errors by which $\beta$ deviates from $1$. If ADF is less than the critical values from the ADF distribution table, we can reject the null hypothesis and conclude that the time series is stationary.

        1. Compare the test statistic to the critical values in the ADF distribution table to determine the level of significance. The critical values depend on the sample size, the level of significance, and the number of lags in the model.

        1. Finally, interpret the results and draw conclusions about the stationarity of the time series. If the null hypothesis is rejected, then the time series is stationary and can be used for forecasting. If the null hypothesis is not rejected, then the time series is non-stationary and requires further pre-processing before it can be used for forecasting.

    ??? note "Notes"
        The null hypothesis of the Augmented Dickey-Fuller is that there is a unit root, with the alternative that there is no unit root. If the p-value is above a critical size, then we cannot reject that there is a unit root.

        The p-values are obtained through regression surface approximation from MacKinnon 1994, but using the updated 2010 tables. If the p-value is close to significant, then the critical values should be used to judge whether to reject the null.

        The `autolag` option and `maxlag` for it are described in Greene.

    ??? success "Credit"
        - All credit goes to the [`statsmodels`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html) library.

    ??? question "References"
        - Baum, C.F. (2004). ZANDREWS: Stata module to calculate Zivot-Andrews unit root test in presence of structural break," Statistical Software Components S437301, Boston College Department of Economics, revised 2015.
        - Schwert, G.W. (1989). Tests for unit roots: A Monte Carlo investigation. Journal of Business & Economic Statistics, 7: 147-159.
        - Zivot, E., and Andrews, D.W.K. (1992). Further evidence on the great crash, the oil-price shock, and the unit-root hypothesis. Journal of Business & Economic Studies, 10: 251-270.

    ??? tip "See Also"
        - [`statsmodels.tsa.stattools.adfuller`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html): Augmented Dickey-Fuller unit root test.
        - [`statsmodels.tsa.stattools.kpss`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html): Kwiatkowski-Phillips-Schmidt-Shin stationarity test.
        - [`statsmodels.tsa.stattools.range_unit_root_test`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.range_unit_root_test.html): Range Unit-Root test.
        - [`statsmodels.tsa.stattools.zivot_andrews`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.zivot_andrews.html): Zivot-Andrews structural break test.
        - [`pmdarima.arima.PPTest`](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.PPTest.html): Phillips-Perron unit root test.
        - [`arch.unitroot.DFGLS`](https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.DFGLS.html): Elliot, Rothenberg and Stock's GLS-detrended Dickey-Fuller.
        - [`arch.unitroot.VarianceRatio`](https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.VarianceRatio.html): Variance Ratio test of a random walk.
        - [`ts_stat_tests.algorithms.stationarity.adf`][ts_stat_tests.algorithms.stationarity.adf]: Augmented Dickey-Fuller unit root test.
        - [`ts_stat_tests.algorithms.stationarity.kpss`][ts_stat_tests.algorithms.stationarity.kpss]: Kwiatkowski-Phillips-Schmidt-Shin stationarity test.
        - [`ts_stat_tests.algorithms.stationarity.rur`][ts_stat_tests.algorithms.stationarity.rur]: Range Unit-Root test of stationarity.
        - [`ts_stat_tests.algorithms.stationarity.za`][ts_stat_tests.algorithms.stationarity.za]: Zivot-Andrews structural break unit root test.
        - [`ts_stat_tests.algorithms.stationarity.pp`][ts_stat_tests.algorithms.stationarity.pp]: Phillips-Perron unit root test.
        - [`ts_stat_tests.algorithms.stationarity.ers`][ts_stat_tests.algorithms.stationarity.ers]: Elliot, Rothenberg and Stock's GLS-detrended Dickey-Fuller test.
        - [`ts_stat_tests.algorithms.stationarity.vr`][ts_stat_tests.algorithms.stationarity.vr]: Variance Ratio test of a random walk.
    """
    res: Any = _adfuller(  # Using `Any` to avoid ty issues with statsmodels stubs
        x=x,
        maxlag=maxlag,
        regression=regression,
        autolag=autolag,  # type: ignore[arg-type] # statsmodels stubs are often missing `None`
        store=store,
        regresults=regresults,
    )

    if store:
        # returns (stat, pval, crit, store)
        return float(res[0]), float(res[1]), dict(res[2]), res[3]

    if autolag is None:
        # returns (stat, pval, lags, nobs, crit)
        return (
            float(res[0]),
            float(res[1]),
            int(res[2]),
            int(res[3]),
            dict(res[4]),
        )

    # returns (stat, pval, lags, nobs, crit, icbest)
    return (
        float(res[0]),
        float(res[1]),
        int(res[2]),
        int(res[3]),
        dict(res[4]),
        float(res[5]),
    )


@overload
def kpss(
    x: ArrayLike,
    regression: VALID_KPSS_REGRESSION_OPTIONS = "c",
    nlags: Optional[Union[VALID_KPSS_NLAGS_OPTIONS, int]] = None,
    *,
    store: Literal[True],
) -> tuple[float, float, int, dict, ResultsStore]: ...
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
    tuple[float, float, int, dict, ResultsStore],
    tuple[float, float, int, dict],
]:
    r"""
    !!! note "Summary"
        Computes the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for the null hypothesis that `x` is level or trend stationary.

    ???+ abstract "Details"

        The Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test is another statistical test used to determine whether a time series is stationary or not. The KPSS test is the opposite of the Augmented Dickey-Fuller (ADF) test, which tests for the presence of a unit root in the time series.

        The KPSS test involves regressing the time series on a constant and a time trend. The null hypothesis of the test is that the time series is stationary. The alternative hypothesis is that the time series has a unit root, which means that it is non-stationary.

        The test statistic is calculated by taking the sum of the squared residuals of the regression. If the test statistic is greater than a critical value at a given significance level, typically 0.05, then we reject the null hypothesis and conclude that the time series is non-stationary. If the test statistic is less than the critical value, then we fail to reject the null hypothesis and conclude that the time series is stationary.

        In practical terms, if a time series is found to be non-stationary by the KPSS test, one can apply differencing to the time series until it becomes stationary. This involves taking the difference between consecutive observations and potentially repeating this process until the time series is stationary.

        Overall, the ADF and KPSS tests are both important tools in time series analysis and forecasting, as they help identify whether a time series is stationary or non-stationary, which can have implications for the choice of forecasting models and methods.

    Params:
        x (ArrayLike):
            The data series to test.
        regression (VALID_KPSS_REGRESSION_OPTIONS, optional):
            The null hypothesis for the KPSS test.

            - `"c"`: The data is stationary around a constant (default).
            - `"ct"`: The data is stationary around a trend.

            Defaults to `"c"`.
        nlags (Optional[Union[VALID_KPSS_NLAGS_OPTIONS, int]], optional):
            Indicates the number of lags to be used.

            - If `"auto"` (default), `lags` is calculated using the data-dependent method of Hobijn et al. (1998). See also Andrews (1991), Newey & West (1994), and Schwert (1989).
            - If set to `"legacy"`, uses $int(12 \\times (\\frac{n}{100})^{\\frac{1}{4}})$, as outlined in Schwert (1989).

            Defaults to `None`.
        store (bool, optional):
            If `True`, then a result instance is returned additionally to the KPSS statistic.<br>
            Defaults to `False`.

    Returns:
        (Union[tuple[float, float, int, dict, ResultsStore], tuple[float, float, int, dict]]):
            Returns a tuple containing:
            - `stat` (float): The KPSS test statistic.
            - `pvalue` (float): The p-value of the test.
            - `lags` (int): The truncation lag parameter.
            - `crit` (dict): The critical values at 10%, 5%, 2.5%, and 1%.
            - `resstore` (Optional[ResultsStore]): Result instance (if `store` is `True`).

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.stationarity import kpss
        >>> from ts_stat_tests.utils.data import data_airline, data_normal
        >>> normal = data_normal
        >>> airline = data_airline.values

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Stationary Series"}
        >>> stat, pvalue, lags, crit = kpss(x=normal)
        >>> print(f"KPSS statistic: {stat:.4f}")
        KPSS statistic: 0.0858
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.1000

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Airline Passengers Data"}
        >>> stat, pvalue, lags, crit = kpss(x=airline)
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.0100

        ```

    ??? equation "Calculation"

        The mathematical equation for the KPSS test for stationarity in time series forecasting is:

        $$
        y_t = \mu_t + \epsilon_t
        $$

        where:

        - $y_t$ is the value of the time series at time $t$.
        - $\mu_t$ is the trend component of the time series.
        - $\epsilon_t$ is the error term.

        The KPSS test involves testing the null hypothesis that the time series is trend stationary, which means that the trend component of the time series is stationary over time. If the null hypothesis is rejected, then the time series is non-stationary and requires further pre-processing before it can be used for forecasting.

        Here are the detailed steps for how to calculate the KPSS test:

        1. Collect your time series data and plot it to visually check for any trends, seasonal patterns, or other patterns that could make the data non-stationary. If you detect any such patterns, you will need to pre-process your data (e.g., detrending, deseasonalizing, etc.) to remove these effects.

        1. Divide your time series data into multiple overlapping windows of equal size. The length of each window depends on the length of your time series and the level of detail you want to capture.

        1. Calculate the trend component $\mu_t$ for each window using a trend estimation method. There are several methods for estimating the trend component, such as the Hodrick-Prescott filter, the Christiano-Fitzgerald filter, or simple linear regression. The choice of method depends on the characteristics of your data and the level of accuracy you want to achieve.

        1. Calculate the residual series $\epsilon_t$ by subtracting the trend component from the original time series:

            $$
            \epsilon_t = y_t - \mu_t
            $$

        1. Estimate the variance of the residual series using a suitable estimator, such as the Newey-West estimator or the Bartlett kernel estimator. This step is necessary to correct for any serial correlation in the residual series.

        1. Calculate the test statistic, which is given by:

            $$
            KPSS = T \times \sum_{t=1}^T \frac {S_t^2} {\sigma^2}
            $$

            where:

            - $T$ is the number of observations in the time series.
            - $S_t$ is the cumulative sum of the residual series up to time $t$, i.e., $S_t = \sum_{i=1}^t \epsilon_i$.
            - $\sigma^2$ is the estimated variance of the residual series.

            The test statistic measures the strength of the trend component relative to the residual series. If KPSS is greater than the critical values from the KPSS distribution table, we can reject the null hypothesis and conclude that the time series is non-stationary.

        1. Finally, interpret the results and draw conclusions about the stationarity of the time series. If the null hypothesis is rejected, then the time series is non-stationary and requires further pre-processing before it can be used for forecasting. If the null hypothesis is not rejected, then the time series is trend stationary and can be used for forecasting.

    ??? note "Notes"
        To estimate $\sigma^2$ the Newey-West estimator is used. If `lags` is `"legacy"`, the truncation lag parameter is set to $int(12 \times (\frac{n}{100})^{\frac{1}{4}})$, as outlined in Schwert (1989). The p-values are interpolated from Table 1 of Kwiatkowski et al. (1992). If the computed statistic is outside the table of critical values, then a warning message is generated.

        Missing values are not handled.

    ??? success "Credit"
        - All credit goes to the [`statsmodels`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html) library.

    ??? question "References"
        - Andrews, D.W.K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. Econometrica, 59: 817-858.
        - Hobijn, B., Frances, B.H., & Ooms, M. (2004). Generalizations of the KPSS-test for stationarity. Statistica Neerlandica, 52: 483-502.
        - Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., & Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. Journal of Econometrics, 54: 159-178.
        - Newey, W.K., & West, K.D. (1994). Automatic lag selection in covariance matrix estimation. Review of Economic Studies, 61: 631-653.
        - Schwert, G. W. (1989). Tests for unit roots: A Monte Carlo investigation. Journal of Business and Economic Statistics, 7 (2): 147-159.

    ??? tip "See Also"
        - [`statsmodels.tsa.stattools.adfuller`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html): Augmented Dickey-Fuller unit root test.
        - [`statsmodels.tsa.stattools.kpss`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html): Kwiatkowski-Phillips-Schmidt-Shin stationarity test.
        - [`statsmodels.tsa.stattools.range_unit_root_test`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.range_unit_root_test.html): Range Unit-Root test.
        - [`statsmodels.tsa.stattools.zivot_andrews`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.zivot_andrews.html): Zivot-Andrews structural break test.
        - [`pmdarima.arima.PPTest`](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.PPTest.html): Phillips-Perron unit root test.
        - [`arch.unitroot.DFGLS`](https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.DFGLS.html): Elliot, Rothenberg and Stock's GLS-detrended Dickey-Fuller.
        - [`arch.unitroot.VarianceRatio`](https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.VarianceRatio.html): Variance Ratio test of a random walk.
        - [`ts_stat_tests.algorithms.stationarity.adf`][ts_stat_tests.algorithms.stationarity.adf]: Augmented Dickey-Fuller unit root test.
        - [`ts_stat_tests.algorithms.stationarity.kpss`][ts_stat_tests.algorithms.stationarity.kpss]: Kwiatkowski-Phillips-Schmidt-Shin stationarity test.
        - [`ts_stat_tests.algorithms.stationarity.rur`][ts_stat_tests.algorithms.stationarity.rur]: Range Unit-Root test of stationarity.
        - [`ts_stat_tests.algorithms.stationarity.za`][ts_stat_tests.algorithms.stationarity.za]: Zivot-Andrews structural break unit root test.
        - [`ts_stat_tests.algorithms.stationarity.pp`][ts_stat_tests.algorithms.stationarity.pp]: Phillips-Perron unit root test.
        - [`ts_stat_tests.algorithms.stationarity.ers`][ts_stat_tests.algorithms.stationarity.ers]: Elliot, Rothenberg and Stock's GLS-detrended Dickey-Fuller test.
        - [`ts_stat_tests.algorithms.stationarity.vr`][ts_stat_tests.algorithms.stationarity.vr]: Variance Ratio test of a random walk.
    """
    _nlags: Union[VALID_KPSS_NLAGS_OPTIONS, int] = nlags if nlags is not None else "auto"
    return _kpss(x=x, regression=regression, nlags=_nlags, store=store)


@overload
def rur(x: ArrayLike, *, store: Literal[True]) -> tuple[float, float, dict, ResultsStore]: ...
@overload
def rur(x: ArrayLike, *, store: Literal[False] = False) -> tuple[float, float, dict]: ...
@typechecked
def rur(x: ArrayLike, *, store: bool = False) -> Union[
    tuple[float, float, dict, ResultsStore],
    tuple[float, float, dict],
]:
    r"""
    !!! note "Summary"
        Computes the Range Unit-Root (RUR) test for the null hypothesis that x is stationary.

    ???+ abstract "Details"

        The Range Unit-Root (RUR) test is a statistical test used to determine whether a time series is stationary or not. It is based on the range of the time series and does not require any knowledge of the underlying stochastic process.

        The RUR test involves dividing the time series into non-overlapping windows of a fixed size and calculating the range of each window. Then, the range of the entire time series is calculated. If the time series is stationary, the range of the entire time series should be proportional to the square root of the window size. If the time series is non-stationary, the range of the entire time series will grow with the window size.

        The null hypothesis of the RUR test is that the time series is non-stationary (unit root). The alternative hypothesis is that the time series is stationary. If the test statistic is less than a critical value at a given significance level, typically 0.05, then we reject the null hypothesis and conclude that the time series is stationary. If the test statistic is greater than the critical value, then we fail to reject the null hypothesis and conclude that the time series is non-stationary.

        In practical terms, if a time series is found to be non-stationary by the RUR test, one can apply differencing to the time series until it becomes stationary. This involves taking the difference between consecutive observations and potentially repeating this process until the time series is stationary.

        The RUR test is a simple and computationally efficient test for stationarity, but it may not be as powerful as other unit root tests in detecting non-stationarity in some cases. It is important to use multiple tests to determine the stationarity of a time series, as no single test is perfect in all situations.

    Params:
        x (ArrayLike):
            The data series to test.
        store (bool, optional):
            If `True`, then a result instance is returned additionally to the RUR statistic.<br>
            Defaults to `False`.

    Returns:
        (Union[tuple[float, float, dict, ResultsStore], tuple[float, float, dict]]):
            Returns a tuple containing:
            - `stat` (float): The RUR test statistic.
            - `pvalue` (float): The p-value of the test.
            - `crit` (dict): The critical values at 10%, 5%, 2.5%, and 1%.
            - `resstore` (Optional[ResultsStore]): Result instance (if `store` is `True`).

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.utils.data import data_airline, data_normal, data_trend, data_sine
        >>> from ts_stat_tests.algorithms.stationarity import rur
        >>> normal = data_normal
        >>> trend = data_trend
        >>> seasonal = data_sine
        >>> airline = data_airline.values

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Stationary Series"}
        >>> stat, pvalue, crit = rur(x=normal)
        >>> print(f"RUR statistic: {stat:.4f}")
        RUR statistic: 0.3479
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.0100

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Trend-Stationary Series"}
        >>> stat, pvalue, crit = rur(x=trend)
        >>> print(f"RUR statistic: {stat:.4f}")
        RUR statistic: 31.5912
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.9500

        ```

        ```pycon {.py .python linenums="1" title="Example 3: Seasonal Series"}
        >>> stat, pvalue, crit = rur(x=seasonal)
        >>> print(f"RUR statistic: {stat:.4f}")
        RUR statistic: 0.9129
        >>> print(f"p-value: {pvalue:.04f}")
        p-value: 0.0100

        ```

        ```pycon {.py .python linenums="1" title="Example 4: Real-World Time Series"}
        >>> stat, pvalue, crit = rur(x=airline)
        >>> print(f"RUR statistic: {stat:.4f}")
        RUR statistic: 2.3333
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.9000

        ```

    ??? equation "Calculation"

        The mathematical equation for the RUR test is:

        $$
        y_t = \rho y_{t-1} + \epsilon_t
        $$

        where:

        - $y_t$ is the value of the time series at time $t$.
        - $\rho$ is the parameter of the unit root process.
        - $y_{t-1}$ is the value of the time series at time $t-1$.
        - $\epsilon_t$ is a stationary error term with mean zero and constant variance.

        The null hypothesis of the RUR test is that the time series is stationary, and the alternative hypothesis is that the time series is non-stationary with a unit root.

        Here are the detailed steps for how to calculate the RUR test:

        1. Collect your time series data and plot it to visually check for any trends, seasonal patterns, or other patterns that could make the data non-stationary. If you detect any such patterns, you will need to pre-process your data (e.g., detrending, deseasonalizing, etc.) to remove these effects.

        1. Estimate the parameter $\rho$ using the ordinary least squares method. This involves regressing $y_t$ on $y_{t-1}$. The estimated equation is:

            $$
            y_t = \alpha + \rho y_{t-1} + \epsilon_t
            $$

            where:

            - $\alpha$ is the intercept.
            - $\epsilon_t$ is the error term.

        1. Calculate the range of the time series, which is the difference between the maximum and minimum values of the time series:

            $$
            R = \max(y_t) - \min(y_t)
            $$

        1. Calculate the expected range of the time series under the null hypothesis of stationarity, which is given by:

            $$
            E(R) = \frac {T - 1} {2 \sqrt{T}}
            $$

            where:

            - $T$ is the sample size.

        1. Calculate the test statistic, which is given by:

            $$
            RUR = \frac {R - E(R)} {E(R)}
            $$

        1. Compare the test statistic to the critical values in the RUR distribution table to determine the level of significance. The critical values depend on the sample size and the level of significance.

        1. Finally, interpret the results and draw conclusions about the stationarity of the time series. If the null hypothesis is rejected, then the time series is non-stationary with a unit root. If the null hypothesis is not rejected, then the time series is stationary.

        In practice, the RUR test is often conducted using software packages such as R, Python, or MATLAB, which automate the estimation of parameters and calculation of the test statistic.

    ??? note "Notes"
        The p-values are interpolated from Table 1 of Aparicio et al. (2006). If the computed statistic is outside the table of critical values, then a warning message is generated.

        Missing values are not handled.

    !!! success "Credit"
        - All credit goes to the [`statsmodels`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.range_unit_root_test.html) library.

    ??? question "References"
        - Aparicio, F., Escribano A., Sipols, A.E. (2006). Range Unit-Root (RUR) tests: robust against nonlinearities, error distributions, structural breaks and outliers. Journal of Time Series Analysis, 27 (4): 545-576.

    ??? tip "See Also"
        - [`statsmodels.tsa.stattools.adfuller`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html): Augmented Dickey-Fuller unit root test.
        - [`statsmodels.tsa.stattools.kpss`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html): Kwiatkowski-Phillips-Schmidt-Shin stationarity test.
        - [`statsmodels.tsa.stattools.range_unit_root_test`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.range_unit_root_test.html): Range Unit-Root test.
        - [`statsmodels.tsa.stattools.zivot_andrews`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.zivot_andrews.html): Zivot-Andrews structural break test.
        - [`pmdarima.arima.PPTest`](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.PPTest.html): Phillips-Perron unit root test.
        - [`arch.unitroot.DFGLS`](https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.DFGLS.html): Elliot, Rothenberg and Stock's GLS-detrended Dickey-Fuller.
        - [`arch.unitroot.VarianceRatio`](https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.VarianceRatio.html): Variance Ratio test of a random walk.
        - [`ts_stat_tests.algorithms.stationarity.adf`][ts_stat_tests.algorithms.stationarity.adf]: Augmented Dickey-Fuller unit root test.
        - [`ts_stat_tests.algorithms.stationarity.kpss`][ts_stat_tests.algorithms.stationarity.kpss]: Kwiatkowski-Phillips-Schmidt-Shin stationarity test.
        - [`ts_stat_tests.algorithms.stationarity.rur`][ts_stat_tests.algorithms.stationarity.rur]: Range Unit-Root test of stationarity.
        - [`ts_stat_tests.algorithms.stationarity.za`][ts_stat_tests.algorithms.stationarity.za]: Zivot-Andrews structural break unit root test.
        - [`ts_stat_tests.algorithms.stationarity.pp`][ts_stat_tests.algorithms.stationarity.pp]: Phillips-Perron unit root test.
        - [`ts_stat_tests.algorithms.stationarity.ers`][ts_stat_tests.algorithms.stationarity.ers]: Elliot, Rothenberg and Stock's GLS-detrended Dickey-Fuller test.
        - [`ts_stat_tests.algorithms.stationarity.vr`][ts_stat_tests.algorithms.stationarity.vr]: Variance Ratio test of a random walk.
    """
    return _rur(x=x, store=store)


@typechecked
def za(
    x: ArrayLike,
    trim: float = 0.15,
    maxlag: Optional[int] = None,
    regression: VALID_ZA_REGRESSION_OPTIONS = "c",
    autolag: Optional[VALID_ZA_AUTOLAG_OPTIONS] = "AIC",
) -> tuple[float, float, dict, int, int]:
    r"""
    !!! note "Summary"
        The Zivot-Andrews (ZA) test tests for a unit root in a univariate process in the presence of serial correlation and a single structural break.

    ???+ abstract "Details"
        The Zivot-Andrews (ZA) test is a statistical test used to determine whether a time series is stationary or not in the presence of structural breaks. Structural breaks refer to significant changes in the underlying stochastic process of the time series, which can cause non-stationarity.

        The ZA test involves running a regression of the time series on a constant and a linear time trend, and testing whether the residuals of the regression are stationary or not. The null hypothesis of the test is that the time series is stationary with a single break point, while the alternative hypothesis is that the time series is non-stationary with a single break point.

        The test statistic is calculated by first estimating the break point using a likelihood ratio test. Then, the test statistic is calculated based on the estimated break point and the residuals of the regression. If the test statistic is greater than a critical value at a given significance level, typically 0.05, then we reject the null hypothesis and conclude that the time series is non-stationary with a structural break. If the test statistic is less than the critical value, then we fail to reject the null hypothesis and conclude that the time series is stationary with a structural break.

        In practical terms, if a time series is found to be non-stationary with a structural break by the ZA test, one can apply methods to account for the structural break, such as including dummy variables in the regression or using time series models that allow for structural breaks.

        Overall, the ZA test is a useful tool in time series analysis and forecasting when there is a suspicion of structural breaks in the data. However, it is important to note that the test may not detect multiple break points or breaks that are not well-separated in time.

    Params:
        x (ArrayLike):
            The data series to test.
        trim (float):
            The percentage of series at begin/end to exclude.
            Default: `0.15`
        maxlag (Optional[int]):
            The maximum lag which is included in test.
            Default: `None`
        regression (VALID_ZA_REGRESSION_OPTIONS):
            Constant and trend order to include in regression.

            - `"c"`: constant only (default).
            - `"t"`: trend only.
            - `"ct"`: constant and trend.

            Default: `"c"`
        autolag (Optional[VALID_ZA_AUTOLAG_OPTIONS]):
            The method to select the lag length.

            - If `None`, then `maxlag` lags are used.
            - If `"AIC"` (default) or `"BIC"`, then the number of lags is chosen.

            Default: `"AIC"`

    Returns:
        (tuple[float, float, dict, int, int]):
            Returns a tuple containing:
            - `zastat` (float): The test statistic.
            - `pvalue` (float): The p-value.
            - `cvdict` (dict): Critical values at the $1\%$, $5\%$, and $10\%$ levels.
            - `baselag` (int): Lags used for period regressions.
            - `pbidx` (int): Break period index.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.utils.data import data_airline, data_normal, data_noise
        >>> from ts_stat_tests.algorithms.stationarity import za
        >>> normal = data_normal
        >>> noise = data_noise
        >>> airline = data_airline.values

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Stationary Series"}
        >>> stat, pvalue, crit, lags, break_idx = za(x=normal)
        >>> print(f"ZA statistic: {stat:.4f}")
        ZA statistic: -30.8800
        >>> print(f"p-value: {pvalue:.4e}")
        p-value: 1.0000e-05

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Noisy Series"}
        >>> stat, pvalue, crit, lags, break_idx = za(x=noise)
        >>> print(f"ZA statistic: {stat:.4f}")
        ZA statistic: -32.4316
        >>> print(f"p-value: {pvalue:.4e}")
        p-value: 1.0000e-05

        ```

        ```pycon {.py .python linenums="1" title="Example 3: Real-World Time Series"}
        >>> stat, pvalue, crit, lags, break_idx = za(x=airline)
        >>> print(f"ZA statistic: {stat:.4f}")
        ZA statistic: -3.6508
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.5808

        ```

    ??? equation "Calculation"

        The mathematical equation for the Zivot-Andrews test is:

        $$
        y_t = \alpha + \beta t + \gamma y_{t-1} + \delta_1 D_t + \delta_2 t D_t + \epsilon_t
        $$

        where:

        - $y_t$ is the value of the time series at time $t$.
        - $\alpha$ is the intercept.
        - $\beta$ is the slope coefficient of the time trend.
        - $\gamma$ is the coefficient of the lagged dependent variable.
        - $D_t$ is a dummy variable that takes a value of 1 after the suspected structural break point, and 0 otherwise.
        - $\delta_1$ and $\delta_2$ are the coefficients of the dummy variable and the interaction term of the dummy variable and time trend, respectively.
        - $\epsilon_t$ is a stationary error term with mean zero and constant variance.

        The null hypothesis of the Zivot-Andrews test is that the time series is non-stationary, and the alternative hypothesis is that the time series is stationary with a single structural break.

        Here are the detailed steps for how to calculate the Zivot-Andrews test:

        1. Collect your time series data and plot it to visually check for any trends, seasonal patterns, or other patterns that could make the data non-stationary. If you detect any such patterns, you will need to pre-process your data (e.g., detrending, deseasonalizing, etc.) to remove these effects.

        1. Estimate the parameters of the model using the least squares method. This involves regressing $y_t$ on $t$, $y_{t-1}$, $D_t$, and $t D_t$. The estimated equation is:

            $$
            y_t = \alpha + \beta t + \gamma y_{t-1} + \delta_1 D_t + \delta_2 t D_t + \epsilon_t
            $$

        1. Perform a unit root test on the residuals to check for stationarity. The most commonly used unit root tests for this purpose are the Augmented Dickey-Fuller (ADF) test and the Phillips-Perron (PP) test.

        1. Calculate the test statistic, which is based on the largest root of the following equation:

            $$
            \Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \delta_1 D_t + \delta_2 t D_t + \epsilon_t
            $$

            where:

            - $\Delta$ is the first difference operator.

        1. Determine the critical values of the test statistic from the Zivot-Andrews distribution table. The critical values depend on the sample size, the level of significance, and the number of lagged dependent variables in the model.

        1. Finally, interpret the results and draw conclusions about the stationarity of the time series. If the null hypothesis is rejected, then the time series is stationary with a structural break. If the null hypothesis is not rejected, then the time series is non-stationary and may require further processing to make it stationary.

        In practice, the Zivot-Andrews test is often conducted using software packages such as R, Python, or MATLAB, which automate the estimation of parameters and calculation of the test statistic.

    ??? note "Notes"
        H0 = unit root with a single structural break

        Algorithm follows Baum (2004/2015) approximation to original Zivot-Andrews method. Rather than performing an autolag regression at each candidate break period (as per the original paper), a single autolag regression is run up-front on the base model (constant + trend with no dummies) to determine the best lag length. This lag length is then used for all subsequent break-period regressions. This results in significant run time reduction but also slightly more pessimistic test statistics than the original Zivot-Andrews method, although no attempt has been made to characterize the size/power trade-off.

    ??? success "Credit"
        - All credit goes to the [`statsmodels`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.zivot_andrews.html) library.

    ??? question "References"
        - Baum, C.F. (2004). ZANDREWS: Stata module to calculate Zivot-Andrews unit root test in presence of structural break," Statistical Software Components S437301, Boston College Department of Economics, revised 2015.
        - Schwert, G.W. (1989). Tests for unit roots: A Monte Carlo investigation. Journal of Business & Economic Statistics, 7: 147-159.
        - Zivot, E., and Andrews, D.W.K. (1992). Further evidence on the great crash, the oil-price shock, and the unit-root hypothesis. Journal of Business & Economic Studies, 10: 251-270.

    ??? tip "See Also"
        - [`statsmodels.tsa.stattools.adfuller`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html): Augmented Dickey-Fuller unit root test.
        - [`statsmodels.tsa.stattools.kpss`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html): Kwiatkowski-Phillips-Schmidt-Shin stationarity test.
        - [`statsmodels.tsa.stattools.range_unit_root_test`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.range_unit_root_test.html): Range Unit-Root test.
        - [`statsmodels.tsa.stattools.zivot_andrews`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.zivot_andrews.html): Zivot-Andrews structural break test.
        - [`pmdarima.arima.PPTest`](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.PPTest.html): Phillips-Perron unit root test.
        - [`arch.unitroot.DFGLS`](https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.DFGLS.html): Elliot, Rothenberg and Stock's GLS-detrended Dickey-Fuller.
        - [`arch.unitroot.VarianceRatio`](https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.VarianceRatio.html): Variance Ratio test of a random walk.
        - [`ts_stat_tests.algorithms.stationarity.adf`][ts_stat_tests.algorithms.stationarity.adf]: Augmented Dickey-Fuller unit root test.
        - [`ts_stat_tests.algorithms.stationarity.kpss`][ts_stat_tests.algorithms.stationarity.kpss]: Kwiatkowski-Phillips-Schmidt-Shin stationarity test.
        - [`ts_stat_tests.algorithms.stationarity.rur`][ts_stat_tests.algorithms.stationarity.rur]: Range Unit-Root test of stationarity.
        - [`ts_stat_tests.algorithms.stationarity.za`][ts_stat_tests.algorithms.stationarity.za]: Zivot-Andrews structural break unit root test.
        - [`ts_stat_tests.algorithms.stationarity.pp`][ts_stat_tests.algorithms.stationarity.pp]: Phillips-Perron unit root test.
        - [`ts_stat_tests.algorithms.stationarity.ers`][ts_stat_tests.algorithms.stationarity.ers]: Elliot, Rothenberg and Stock's GLS-detrended Dickey-Fuller test.
        - [`ts_stat_tests.algorithms.stationarity.vr`][ts_stat_tests.algorithms.stationarity.vr]: Variance Ratio test of a random walk.
    """
    res: Any = _za(
        x=x,
        trim=trim,
        maxlag=maxlag,
        regression=regression,
        autolag=autolag,  # type: ignore[arg-type] # statsmodels stubs are often missing None
    )
    return (
        float(res[0]),
        float(res[1]),
        dict(res[2]),
        int(res[3]),
        int(res[4]),
    )


@typechecked
def pp(
    x: ArrayLike,
    lags: Optional[int] = None,
    trend: VALID_PP_TREND_OPTIONS = "c",
    test_type: VALID_PP_TEST_TYPE_OPTIONS = "tau",
) -> tuple[float, float, int, dict]:
    r"""
    !!! note "Summary"
        Conduct a Phillips-Perron (PP) test for stationarity.

        In statistics, the Phillips-Perron test (named after Peter C. B. Phillips and Pierre Perron) is a unit root test. It is used in time series analysis to test the null hypothesis that a time series is integrated of order $1$. It builds on the Dickey-Fuller test of the null hypothesis $p=0$.

    ???+ abstract "Details"

        The Phillips-Perron (PP) test is a statistical test used to determine whether a time series is stationary or not. It is similar to the Augmented Dickey-Fuller (ADF) test, but it has some advantages, especially in the presence of autocorrelation and heteroscedasticity.

        The PP test involves regressing the time series on a constant and a linear time trend, and testing whether the residuals of the regression are stationary or not. The null hypothesis of the test is that the time series is non-stationary, while the alternative hypothesis is that the time series is stationary.

        The test statistic is calculated by taking the sum of the squared residuals of the regression, which is adjusted for autocorrelation and heteroscedasticity. The PP test also accounts for the bias in the standard errors of the test statistic, which can lead to incorrect inference in small samples.

        If the test statistic is less than a critical value at a given significance level, typically 0.05, then we reject the null hypothesis and conclude that the time series is stationary. If the test statistic is greater than the critical value, then we fail to reject the null hypothesis and conclude that the time series is non-stationary.

        In practical terms, if a time series is found to be non-stationary by the PP test, one can apply differencing to the time series until it becomes stationary. This involves taking the difference between consecutive observations and potentially repeating this process until the time series is stationary.

        Overall, the PP test is a powerful and robust test for stationarity, and it is widely used in time series analysis and forecasting. However, it is important to use multiple tests and diagnostic tools to determine the stationarity of a time series, as no single test is perfect in all situations.

    Params:
        x (ArrayLike):
            The data series to test.
        lags (Optional[int], optional):
            The number of lags to use in the Newey-West estimator of the variance. If omitted or `None`, the lag length is selected automatically.<br>
            Defaults to `None`.
        trend (VALID_PP_TREND_OPTIONS, optional):
            The trend component to include in the test.

            - `"n"`: No constant, no trend.
            - `"c"`: Include a constant (default).
            - `"ct"`: Include a constant and linear time trend.

            Defaults to `"c"`.
        test_type (VALID_PP_TEST_TYPE_OPTIONS, optional):
            The type of test statistic to compute:

            - `"tau"`: The t-statistic based on the augmented regression (default).
            - `"rho"`: The normalized autocorrelation coefficient (also known as the $Z(\\alpha)$ test).

            Defaults to `"tau"`.

    Returns:
        (tuple[float, float, int, dict]):
            Returns a tuple containing:
            - `stat` (float): The test statistic.
            - `pvalue` (float): The p-value for the test statistic.
            - `lags` (int): The number of lags used in the test.
            - `crit` (dict): The critical values at 1%, 5%, and 10%.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.stationarity import pp
        >>> from ts_stat_tests.utils.data import data_airline, data_normal, data_trend, data_sine
        >>> normal = data_normal
        >>> trend = data_trend
        >>> seasonal = data_sine
        >>> airline = data_airline.values

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Stationary Series"}
        >>> stat, pvalue, lags, crit = pp(x=normal)
        >>> print(f"PP statistic: {stat:.4f}")
        PP statistic: -30.7758
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.0000

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Trend-Stationary Series"}
        >>> stat, pvalue, lags, crit = pp(x=trend, trend="ct")
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.0000

        ```

        ```pycon {.py .python linenums="1" title="Example 3: Seasonal Series"}
        >>> stat, pvalue, lags, crit = pp(x=seasonal)
        >>> print(f"PP statistic: {stat:.4f}")
        PP statistic: -8.0571
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.0000

        ```

        ```pycon {.py .python linenums="1" title="Example 4: Real-World Time Series"}
        >>> stat, pvalue, lags, crit = pp(x=airline)
        >>> print(f"PP statistic: {stat:.4f}")
        PP statistic: -1.3511
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.6055

        ```

        ```pycon {.py .python linenums="1" title="Example 5: PP test with excessive lags (coverage check)"}
        >>> from ts_stat_tests.algorithms.stationarity import pp
        >>> from ts_stat_tests.utils.data import data_normal
        >>> # data_normal has 1000 observations. Force lags = 1000 to trigger adjustment.
        >>> res = pp(data_normal, lags=1000)
        >>> print(f"stat: {res[0]:.4f}, lags: {res[2]}")
        stat: -43.6895, lags: 998

        ```

    ??? equation "Calculation"

        The Phillips-Perron (PP) test is a commonly used test for stationarity in time series forecasting. The mathematical equation for the PP test is:

        $$
        y_t = \delta + \pi t + \rho y_{t-1} + \epsilon_t
        $$

        where:

        - $y_t$ is the value of the time series at time $t$.
        - $\delta$ is a constant term.
        - $\pi$ is a coefficient that captures the trend in the data.
        - $\rho$ is a coefficient that captures the autocorrelation in the data.
        - $y_{t-1}$ is the lagged value of the time series at time $t-1$.
        - $\epsilon_t$ is a stationary error term with mean zero and constant variance.

        The PP test is based on the idea that if the time series is stationary, then the coefficient $\rho$ should be equal to zero. Therefore, the null hypothesis of the PP test is that the time series is stationary, and the alternative hypothesis is that the time series is non-stationary with a non-zero value of $\rho$.

        Here are the detailed steps for how to calculate the PP test:

        1. Collect your time series data and plot it to visually check for any trends, seasonal patterns, or other patterns that could make the data non-stationary. If you detect any such patterns, you will need to pre-process your data (e.g., detrending, deseasonalizing, etc.) to remove these effects.

        1. Estimate the regression model by regressing $y_t$ on a constant, a linear trend, and the lagged value of $y_{t-1}$. The regression equation is:

            $$
            y_t = \delta + \pi t + \rho y_{t-1} + \epsilon_t
            $$

        1. Calculate the test statistic, which is based on the following equation:

            $$
            z = \left( T^{-\frac{1}{2}} \right) \times \left( \sum_{t=1}^T \left( y_t - \delta - \pi t - \rho y_{t-1} \right) - \left( \frac{1}{T} \right) \times \sum_{t=1}^T \sum_{s=1}^T K \left( \frac{s-t}{h} \right) (y_s - \delta - \pi s - \rho y_{s-1}) \right)
            $$

            where:

            - $T$ is the sample size.
            - $K(\dots)$ is the kernel function, which determines the weight of each observation in the smoothed series. The choice of the kernel function depends on the degree of serial correlation in the data. Typically, a Gaussian kernel or a Bartlett kernel is used.
            - $h$ is the bandwidth parameter, which controls the degree of smoothing of the series. The optimal value of $h$ depends on the sample size and the noise level of the data.

        1. Determine the critical values of the test statistic from the PP distribution table. The critical values depend on the sample size and the level of significance.

        1. Finally, interpret the results and draw conclusions about the stationarity of the time series. If the null hypothesis is rejected, then the time series is non-stationary with a non-zero value of $\rho$. If the null hypothesis is not rejected, then the time series is stationary.

        In practice, the PP test is often conducted using software packages such as R, Python, or MATLAB, which automate the estimation of the regression model and calculation of the test statistic.

    ??? note "Notes"
        This test is generally used indirectly via the [`pmdarima.arima.ndiffs()`](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.ndiffs.html) function, which computes the differencing term, `d`.

        The R code allows for two types of tests: `'Z(alpha)'` and `'Z(t_alpha)'`. Since sklearn does not allow extraction of std errors from the linear model fit, `t_alpha` is much more difficult to achieve, so we do not allow that variant.

    !!! success "Credit"
        - All credit goes to the [`arch`](https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.PhillipsPerron.html) library.

    ??? question "References"
        - Phillips, P. C. B.; Perron, P. (1988). Testing for a Unit Root in Time Series Regression. Biometrika. 75 (2): 335-346.

    ??? tip "See Also"
        - [`statsmodels.tsa.stattools.adfuller`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html): Augmented Dickey-Fuller unit root test.
        - [`statsmodels.tsa.stattools.kpss`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html): Kwiatkowski-Phillips-Schmidt-Shin stationarity test.
        - [`statsmodels.tsa.stattools.range_unit_root_test`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.range_unit_root_test.html): Range Unit-Root test.
        - [`statsmodels.tsa.stattools.zivot_andrews`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.zivot_andrews.html): Zivot-Andrews structural break test.
        - [`pmdarima.arima.PPTest`](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.PPTest.html): Phillips-Perron unit root test.
        - [`arch.unitroot.DFGLS`](https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.DFGLS.html): Elliot, Rothenberg and Stock's GLS-detrended Dickey-Fuller.
        - [`arch.unitroot.VarianceRatio`](https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.VarianceRatio.html): Variance Ratio test of a random walk.
        - [`ts_stat_tests.algorithms.stationarity.adf`][ts_stat_tests.algorithms.stationarity.adf]: Augmented Dickey-Fuller unit root test.
        - [`ts_stat_tests.algorithms.stationarity.kpss`][ts_stat_tests.algorithms.stationarity.kpss]: Kwiatkowski-Phillips-Schmidt-Shin stationarity test.
        - [`ts_stat_tests.algorithms.stationarity.rur`][ts_stat_tests.algorithms.stationarity.rur]: Range Unit-Root test of stationarity.
        - [`ts_stat_tests.algorithms.stationarity.za`][ts_stat_tests.algorithms.stationarity.za]: Zivot-Andrews structural break unit root test.
        - [`ts_stat_tests.algorithms.stationarity.pp`][ts_stat_tests.algorithms.stationarity.pp]: Phillips-Perron unit root test.
        - [`ts_stat_tests.algorithms.stationarity.ers`][ts_stat_tests.algorithms.stationarity.ers]: Elliot, Rothenberg and Stock's GLS-detrended Dickey-Fuller test.
        - [`ts_stat_tests.algorithms.stationarity.vr`][ts_stat_tests.algorithms.stationarity.vr]: Variance Ratio test of a random walk.
    """
    _x = np.asarray(x)
    nobs = _x.shape[0]
    _lags = lags
    if _lags is None:
        _lags = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))

    # arch PP test requires lags < nobs-1
    if _lags >= nobs - 1:
        _lags = max(0, nobs - 2)

    res = _pp(y=_x, lags=_lags, trend=trend, test_type=test_type)
    return (float(res.stat), float(res.pvalue), int(res.lags), dict(res.critical_values))


@typechecked
def ers(
    y: ArrayLike,
    lags: Optional[int] = None,
    trend: VALID_ERS_TREND_OPTIONS = "c",
    max_lags: Optional[int] = None,
    method: VALID_ERS_METHOD_OPTIONS = "aic",
    low_memory: Optional[bool] = None,
) -> tuple[float, float, int, dict]:
    r"""
    !!! note "Summary"
        Elliott, Rothenberg and Stock's GLS detrended Dickey-Fuller.

    ???+ abstract "Details"

        The Elliott-Rothenberg-Stock (ERS) test is a statistical test used to determine whether a time series is stationary or not. It is a robust test that is able to handle a wide range of non-stationary processes, including ones with structural breaks, heteroscedasticity, and autocorrelation.

        The ERS test involves fitting a local-to-zero regression of the time series on a constant and a linear time trend, using a kernel function to weight the observations. The test statistic is then calculated based on the sum of the squared residuals of the local-to-zero regression, which is adjusted for the bandwidth of the kernel function and for the correlation of the residuals.

        If the test statistic is less than a critical value at a given significance level, typically 0.05, then we reject the null hypothesis and conclude that the time series is stationary. If the test statistic is greater than the critical value, then we fail to reject the null hypothesis and conclude that the time series is non-stationary.

        In practical terms, if a time series is found to be non-stationary by the ERS test, one can apply differencing to the time series until it becomes stationary. This involves taking the difference between consecutive observations and potentially repeating this process until the time series is stationary.

        Overall, the ERS test is a powerful and flexible test for stationarity, and it is widely used in time series analysis and forecasting. However, it is important to use multiple tests and diagnostic tools to determine the stationarity of a time series, as no single test is perfect in all situations.

    Params:
        y (ArrayLike):
            The data to test for a unit root.
        lags (Optional[int], optional):
            The number of lags to use in the ADF regression. If omitted or `None`, method is used to automatically select the lag length with no more than `max_lags` are included.<br>
            Defaults to `None`.
        trend (VALID_ERS_TREND_OPTIONS, optional):
            The trend component to include in the test

            - `"c"`: Include a constant (Default)
            - `"ct"`: Include a constant and linear time trend

            Defaults to `"c"`.
        max_lags (Optional[int], optional):
            The maximum number of lags to use when selecting lag length. When using automatic lag length selection, the lag is selected using OLS detrending rather than GLS detrending.<br>
            Defaults to `None`.
        method (VALID_ERS_METHOD_OPTIONS, optional):
            The method to use when selecting the lag length

            - `"AIC"`: Select the minimum of the Akaike IC
            - `"BIC"`: Select the minimum of the Schwarz/Bayesian IC
            - `"t-stat"`: Select the minimum of the Schwarz/Bayesian IC

            Defaults to `"aic"`.
        low_memory (Optional[bool], optional):
            Flag indicating whether to use the low-memory algorithm for lag-length selection.
            Defaults to `None`.

    Returns:
        (tuple[float, float, int, dict]):
            Returns a tuple containing:
            - `stat` (float): The test statistic for a unit root.
            - `pvalue` (float): The p-value for the test statistic.
            - `lags` (int): The number of lags used in the test.
            - `crit` (dict): The critical values for the test statistic at the 1%, 5%, and 10% levels.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.stationarity import ers
        >>> from ts_stat_tests.utils.data import data_airline, data_normal, data_noise
        >>> normal = data_normal
        >>> noise = data_noise
        >>> airline = data_airline.values

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Stationary Series"}
        >>> stat, pvalue, lags, crit = ers(y=normal)
        >>> print(f"ERS statistic: {stat:.4f}")
        ERS statistic: -30.1517
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.0000

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Noisy Series"}
        >>> stat, pvalue, lags, crit = ers(y=noise)
        >>> print(f"ERS statistic: {stat:.4f}")
        ERS statistic: -12.6897
        >>> print(f"p-value: {pvalue:.4e}")
        p-value: 1.0956e-21

        ```

        ```pycon {.py .python linenums="1" title="Example 3: Real-World Time Series"}
        >>> stat, pvalue, lags, crit = ers(y=airline)
        >>> print(f"ERS statistic: {stat:.4f}")
        ERS statistic: 0.9918
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.9232

        ```

    ??? equation "Calculation"

        The mathematical equation for the ERS test is:

        $$
        y_t = \mu_t + \epsilon_t
        $$

        where:

        - $y_t$ is the value of the time series at time $t$.
        - $\mu_t$ is a time-varying mean function.
        - $\epsilon_t$ is a stationary error term with mean zero and constant variance.

        The ERS test is based on the idea that if the time series is stationary, then the mean function should be a constant over time. Therefore, the null hypothesis of the ERS test is that the time series is non-stationary (unit root), and the alternative hypothesis is that the time series is stationary.

        Here are the detailed steps for how to calculate the ERS test:

        1. Collect your time series data and plot it to visually check for any trends, seasonal patterns, or other patterns that could make the data non-stationary. If you detect any such patterns, you will need to pre-process your data (e.g., detrending, deseasonalizing, etc.) to remove these effects.

        1. Estimate the time-varying mean function using a local polynomial regression method. The choice of the polynomial degree depends on the complexity of the mean function and the sample size. Typically, a quadratic or cubic polynomial is used. The estimated mean function is denoted as $\mu_t$.

        1. Calculate the test statistic, which is based on the following equation:

            $$
            z = \left( \frac {T-1} {( \frac {1} {12\pi^2 \times \Delta^2} )} \right) ^{\frac{1}{2}} \times \left( \sum_{t=1}^T \frac {(y_t - \mu_t)^2} {T-1} \right)
            $$

            where:

            - $T$ is the sample size
            - $\Delta$ is the bandwidth parameter, which controls the degree of smoothing of the mean function. The optimal value of $\Delta$ depends on the sample size and the noise level of the data.
            - $\pi$ is the constant pi.

        1. Determine the critical values of the test statistic from the ERS distribution table. The critical values depend on the sample size and the level of significance.

        1. Finally, interpret the results and draw conclusions about the stationarity of the time series. If the null hypothesis is rejected, then the time series is non-stationary with a time-varying mean function. If the null hypothesis is not rejected, then the time series is stationary.

        In practice, the ERS test is often conducted using software packages such as R, Python, or MATLAB, which automate the estimation of the time-varying mean function and calculation of the test statistic.

    ??? note "Notes"
        The null hypothesis of the Dickey-Fuller GLS is that there is a unit root, with the alternative that there is no unit root. If the p-value is above a critical size, then the null cannot be rejected and the series appears to be a unit root.

        DFGLS differs from the ADF test in that an initial GLS detrending step is used before a trend-less ADF regression is run.

        Critical values and p-values when trend is `"c"` are identical to the ADF. When trend is set to `"ct"`, they are from Elliott, Rothenberg, and Stock (1996).

    !!! success "Credit"
        - All credit goes to the [`arch`](https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.DFGLS.html) library.

    ??? question "References"
        - Elliott, G. R., T. J. Rothenberg, and J. H. Stock. 1996. Efficient bootstrap for an autoregressive unit root. Econometrica 64: 813-836.
        - Perron, P., & Qu, Z. (2007). A simple modification to improve the finite sample properties of Ng and Perron’s unit root tests. Economics letters, 94(1), 12-19.

    ??? tip "See Also"
        - [`statsmodels.tsa.stattools.adfuller`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html): Augmented Dickey-Fuller unit root test.
        - [`statsmodels.tsa.stattools.kpss`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html): Kwiatkowski-Phillips-Schmidt-Shin stationarity test.
        - [`statsmodels.tsa.stattools.range_unit_root_test`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.range_unit_root_test.html): Range Unit-Root test.
        - [`statsmodels.tsa.stattools.zivot_andrews`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.zivot_andrews.html): Zivot-Andrews structural break test.
        - [`pmdarima.arima.PPTest`](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.PPTest.html): Phillips-Perron unit root test.
        - [`arch.unitroot.DFGLS`](https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.DFGLS.html): Elliot, Rothenberg and Stock's GLS-detrended Dickey-Fuller.
        - [`arch.unitroot.VarianceRatio`](https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.VarianceRatio.html): Variance Ratio test of a random walk.
        - [`ts_stat_tests.algorithms.stationarity.adf`][ts_stat_tests.algorithms.stationarity.adf]: Augmented Dickey-Fuller unit root test.
        - [`ts_stat_tests.algorithms.stationarity.kpss`][ts_stat_tests.algorithms.stationarity.kpss]: Kwiatkowski-Phillips-Schmidt-Shin stationarity test.
        - [`ts_stat_tests.algorithms.stationarity.rur`][ts_stat_tests.algorithms.stationarity.rur]: Range Unit-Root test of stationarity.
        - [`ts_stat_tests.algorithms.stationarity.za`][ts_stat_tests.algorithms.stationarity.za]: Zivot-Andrews structural break unit root test.
        - [`ts_stat_tests.algorithms.stationarity.pp`][ts_stat_tests.algorithms.stationarity.pp]: Phillips-Perron unit root test.
        - [`ts_stat_tests.algorithms.stationarity.ers`][ts_stat_tests.algorithms.stationarity.ers]: Elliot, Rothenberg and Stock's GLS-detrended Dickey-Fuller test.
        - [`ts_stat_tests.algorithms.stationarity.vr`][ts_stat_tests.algorithms.stationarity.vr]: Variance Ratio test of a random walk.
    """
    res = _ers(
        y=np.asarray(y),
        lags=lags,
        trend=trend,
        max_lags=max_lags,
        method=method,
        low_memory=low_memory,
    )
    return (float(res.stat), float(res.pvalue), int(res.lags), dict(res.critical_values))


@typechecked
def vr(
    y: ArrayLike,
    lags: int = 2,
    trend: VALID_VR_TREND_OPTIONS = "c",
    debiased: bool = True,
    robust: bool = True,
    overlap: bool = True,
) -> tuple[float, float, float]:
    r"""
    !!! note "Summary"
        Variance Ratio test of a random walk.

    ???+ abstract "Details"

        The Variance Ratio (VR) test is a statistical test used to determine whether a time series is stationary or not based on the presence of long-term dependence in the series. It is a non-parametric test that can be used to test for the presence of a unit root or a trend in the series.

        The VR test involves calculating the ratio of the variance of the differences of the logarithms of the time series over different time intervals. The variance of the differences of the logarithms is a measure of the volatility of the series, and the ratio of the variances over different intervals is a measure of the long-term dependence in the series.

        If the series is stationary, then the variance ratio will be close to one for all intervals. If the series is non-stationary, then the variance ratio will tend to increase as the length of the interval increases, reflecting the presence of long-term dependence in the series.

        The VR test involves comparing the observed variance ratio to the distribution of variance ratios expected under the null hypothesis of a random walk (non-stationary). If the test statistic is less than a critical value at a given significance level, typically 0.05, then we reject the null hypothesis and conclude that the time series is stationary. If the test statistic is greater than the critical value, then we fail to reject the null hypothesis and conclude that the time series is non-stationary.

        In practical terms, if a time series is found to be non-stationary by the VR test, one can apply differencing to the time series until it becomes stationary. This involves taking the difference between consecutive observations and potentially repeating this process until the time series is stationary.

        Overall, the VR test is a useful and relatively simple test for stationarity that can be applied to a wide range of time series. However, it is important to use multiple tests and diagnostic tools to confirm the stationarity of a time series, as no single test is perfect in all situations.

    Params:
        y (ArrayLike):
            The data to test for a random walk.
        lags (int):
            The number of periods to used in the multi-period variance, which is the numerator of the test statistic. Must be at least 2.<br>
            Defaults to `2`.
        trend (VALID_VR_TREND_OPTIONS, optional):
            `"c"` allows for a non-zero drift in the random walk, while `"n"` requires that the increments to `y` are mean `0`.<br>
            Defaults to `"c"`.
        debiased (bool, optional):
            Indicates whether to use a debiased version of the test. Only applicable if `overlap` is `True`.<br>
            Defaults to `True`.
        robust (bool, optional):
            Indicates whether to use heteroskedasticity robust inference.<br>
            Defaults to `True`.
        overlap (bool, optional):
            Indicates whether to use all overlapping blocks. If `False`, the number of observations in $y-1$ must be an exact multiple of `lags`. If this condition is not satisfied, some values at the end of `y` will be discarded.<br>
            Defaults to `True`.

    Returns:
        (tuple[float, float, float]):
            Returns a tuple containing:
            - `stat` (float): The test statistic for a unit root.
            - `pvalue` (float): The p-value for the test statistic.
            - `vr` (float): The ratio of the long block lags-period variance.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.stationarity import vr
        >>> from ts_stat_tests.utils.data import data_airline, data_normal, data_noise, data_sine
        >>> normal = data_normal
        >>> noise = data_noise
        >>> seasonal = data_sine
        >>> airline = data_airline.values

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Stationary Series"}
        >>> stat, pvalue, variance_ratio = vr(y=normal)
        >>> print(f"VR statistic: {stat:.4f}")
        VR statistic: -12.8518
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.0000
        >>> print(f"Variance ratio: {variance_ratio:.4f}")
        Variance ratio: 0.5202

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Noisy Series"}
        >>> stat, pvalue, variance_ratio = vr(y=noise)
        >>> print(f"VR statistic: {stat:.4f}")
        VR statistic: -11.5007
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.0000
        >>> print(f"Variance ratio: {variance_ratio:.4f}")
        Variance ratio: 0.5094

        ```

        ```pycon {.py .python linenums="1" title="Example 3: Seasonal Series"}
        >>> stat, pvalue, variance_ratio = vr(y=seasonal)
        >>> print(f"VR statistic: {stat:.4f}")
        VR statistic: 44.7019
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.0000
        >>> print(f"Variance ratio: {variance_ratio:.4f}")
        Variance ratio: 1.9980

        ```

        ```pycon {.py .python linenums="1" title="Example 4: Real-World Time Series"}
        >>> stat, pvalue, variance_ratio = vr(y=airline)
        >>> print(f"VR statistic: {stat:.4f}")
        VR statistic: 3.1511
        >>> print(f"p-value: {pvalue:.4f}")
        p-value: 0.0016
        >>> print(f"Variance ratio: {variance_ratio:.4f}")
        Variance ratio: 1.3163

        ```

    ??? equation "Calculation"

        The Variance Ratio (VR) test is a statistical test for stationarity in time series forecasting that is based on the idea that if the time series is stationary, then the variance of the returns should be constant over time. The mathematical equation for the VR test is:

        $$
        VR(k) = \frac {\sigma^2(k)} {k\sigma^2(1)}
        $$

        where:

        - $VR(k)$ is the variance ratio for the time series over $k$ periods.
        - $\sigma^2(k)$ is the variance of the returns over $k$ periods.
        - $\sigma^2(1)$ is the variance of the returns over $1$ period.

        The VR test involves comparing the variance ratio to a critical value, which is derived from the null distribution of the variance ratio under the assumption of a random walk with drift.

        Here are the detailed steps for how to calculate the VR test:

        1. Collect your time series data and compute the log returns, which are defined as:

            $$
            r_t = \log(y_t) - \log(y_{t-1})
            $$

            where:

            - $y_t$ is the value of the time series at time $t$.

        1. Compute the variance of the returns over $k$ periods, which is defined as:

            $$
            \sigma^2(k) = \left( \frac {1} {n-k} \right) \times \sum_{t=k+1}^n (r_t - \mu_k)^2
            $$

            where:

            - $n$ is the sample size.
            - $\mu_k$ is the mean of the returns over $k$ periods, which is defined as:

                $\mu_k = \left( \frac{1} {n-k} \right) \times \sum_{t=k+1}^n r_t$

        1. Compute the variance of the returns over $1$ period, which is defined as:

            $$
            \sigma^2(1) = \left( \frac{1} {n-1} \right) \times \sum_{t=2}^n (r_t - \mu_1)^2
            $$

            where:

            - $\mu_1$ is the mean of the returns over $1$ period, which is defined as:

                $\mu_1 = \left( \frac{1} {n-1} \right) \times \sum_{t=2}^n r_t$

        1. Compute the variance ratio for each value of $k$, which is defined as:

            $$
            VR(k) = \frac {\sigma^2(k)} {k\sigma^2(1)}
            $$

        1. Determine the critical values of the variance ratio from the null distribution table of the VR test, which depend on the sample size, the level of significance, and the lag length $k$.

        1. Finally, compare the variance ratio to the critical value. If the variance ratio is greater than the critical value, then the null hypothesis of a random walk with drift is rejected, and the time series is considered stationary. If the variance ratio is less than or equal to the critical value, then the null hypothesis cannot be rejected, and the time series is considered non-stationary.

        In practice, the VR test is often conducted using software packages such as R, Python, or MATLAB, which automate the calculation of the variance ratio and the determination of the critical value.

    ??? note "Notes"
        The null hypothesis of a VR is that the process is a random walk, possibly plus drift. Rejection of the null with a positive test statistic indicates the presence of positive serial correlation in the time series.

    !!! success "Credit"
        - All credit goes to the [`arch`](https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.VarianceRatio.html) library.

    ??? question "References"
        - Campbell, John Y., Lo, Andrew W. and MacKinlay, A. Craig. (1997) The Econometrics of Financial Markets. Princeton, NJ: Princeton University Press.

    ??? tip "See Also"
        - [`statsmodels.tsa.stattools.adfuller`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html): Augmented Dickey-Fuller unit root test.
        - [`statsmodels.tsa.stattools.kpss`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html): Kwiatkowski-Phillips-Schmidt-Shin stationarity test.
        - [`statsmodels.tsa.stattools.range_unit_root_test`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.range_unit_root_test.html): Range Unit-Root test.
        - [`statsmodels.tsa.stattools.zivot_andrews`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.zivot_andrews.html): Zivot-Andrews structural break test.
        - [`pmdarima.arima.PPTest`](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.PPTest.html): Phillips-Perron unit root test.
        - [`arch.unitroot.DFGLS`](https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.DFGLS.html): Elliot, Rothenberg and Stock's GLS-detrended Dickey-Fuller.
        - [`arch.unitroot.VarianceRatio`](https://arch.readthedocs.io/en/latest/unitroot/generated/arch.unitroot.VarianceRatio.html): Variance Ratio test of a random walk.
        - [`ts_stat_tests.algorithms.stationarity.adf`][ts_stat_tests.algorithms.stationarity.adf]: Augmented Dickey-Fuller unit root test.
        - [`ts_stat_tests.algorithms.stationarity.kpss`][ts_stat_tests.algorithms.stationarity.kpss]: Kwiatkowski-Phillips-Schmidt-Shin stationarity test.
        - [`ts_stat_tests.algorithms.stationarity.rur`][ts_stat_tests.algorithms.stationarity.rur]: Range Unit-Root test of stationarity.
        - [`ts_stat_tests.algorithms.stationarity.za`][ts_stat_tests.algorithms.stationarity.za]: Zivot-Andrews structural break unit root test.
        - [`ts_stat_tests.algorithms.stationarity.pp`][ts_stat_tests.algorithms.stationarity.pp]: Phillips-Perron unit root test.
        - [`ts_stat_tests.algorithms.stationarity.ers`][ts_stat_tests.algorithms.stationarity.ers]: Elliot, Rothenberg and Stock's GLS-detrended Dickey-Fuller test.
        - [`ts_stat_tests.algorithms.stationarity.vr`][ts_stat_tests.algorithms.stationarity.vr]: Variance Ratio test of a random walk.
    """
    res = _vr(
        y=np.asarray(y),
        lags=lags,
        trend=trend,
        debiased=debiased,
        robust=robust,
        overlap=overlap,
    )
    return float(res.stat), float(res.pvalue), float(res.vr)
