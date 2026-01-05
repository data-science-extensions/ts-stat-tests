# ============================================================================ #
#                                                                              #
#     Title: Correlation                                                       #
#     Purpose: Algorithms for Correlation Measures in Time Series Analysis     #
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
    The correlation algorithms module provides functions to compute correlation measures for time series data, including the autocorrelation function (ACF), partial autocorrelation function (PACF), and cross-correlation function (CCF). These measures help identify relationships and dependencies between time series variables, which are essential for time series analysis and forecasting.

    This module leverages the `statsmodels` library to implement these correlation measures, ensuring robust and efficient computations. The functions are designed to handle various input scenarios and provide options for customization, such as specifying the number of lags, confidence intervals, and handling missing data.

    By using these correlation algorithms, users can gain insights into the temporal dependencies within their time series data, aiding in model selection and improving forecasting accuracy.
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
import pandas as pd
from numpy.typing import ArrayLike
from statsmodels.regression.linear_model import (
    RegressionResults,
    RegressionResultsWrapper,
)
from statsmodels.stats.api import (
    acorr_breusch_godfrey,
    acorr_ljungbox,
    acorr_lm,
)
from statsmodels.stats.diagnostic import ResultsStore
from statsmodels.tsa.api import (
    acf as st_acf,
    ccf as st_ccf,
    pacf as st_pacf,
)
from statsmodels.tsa.stattools import ArrayLike1D
from typeguard import typechecked


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["acf", "pacf", "ccf", "lb", "lm", "bglm"]


## --------------------------------------------------------------------------- #
##  Constants                                                               ####
## --------------------------------------------------------------------------- #


VALID_ACF_MISSING_OPTIONS = Literal["none", "raise", "conservative", "drop"]


VALID_PACF_METHOD_OPTIONS = Literal[
    "yw",
    "ywadjusted",
    "ols",
    "ols-inefficient",
    "ols-adjusted",
    "ywm",
    "ywmle",
    "ld",
    "ldadjusted",
    "ldb",
    "ldbiased",
    "burg",
]


VALID_LM_COV_TYPE_OPTIONS = Literal["HC0", "HC1", "HC2", "HC3"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Algorithms                                                             ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@overload
def acf(
    x: ArrayLike,
    adjusted: bool = False,
    nlags: Optional[int] = None,
    fft: bool = True,
    bartlett_confint: bool = True,
    missing: VALID_ACF_MISSING_OPTIONS = "none",
    *,
    qstat: Literal[False] = False,
    alpha: None = None,
) -> np.ndarray: ...
@overload
def acf(
    x: ArrayLike,
    adjusted: bool = False,
    nlags: Optional[int] = None,
    fft: bool = True,
    bartlett_confint: bool = True,
    missing: VALID_ACF_MISSING_OPTIONS = "none",
    *,
    qstat: Literal[False] = False,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]: ...
@overload
def acf(
    x: ArrayLike,
    adjusted: bool = False,
    nlags: Optional[int] = None,
    fft: bool = True,
    bartlett_confint: bool = True,
    missing: VALID_ACF_MISSING_OPTIONS = "none",
    *,
    qstat: Literal[True],
    alpha: None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
@overload
def acf(
    x: ArrayLike,
    adjusted: bool = False,
    nlags: Optional[int] = None,
    fft: bool = True,
    bartlett_confint: bool = True,
    missing: VALID_ACF_MISSING_OPTIONS = "none",
    *,
    qstat: Literal[True],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
@typechecked
def acf(
    x: ArrayLike,
    adjusted: bool = False,
    nlags: Optional[int] = None,
    fft: bool = True,
    bartlett_confint: bool = True,
    missing: VALID_ACF_MISSING_OPTIONS = "none",
    *,
    qstat: bool = False,
    alpha: Optional[float] = None,
) -> Union[
    np.ndarray,
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    r"""
    !!! note "Summary"

        The autocorrelation function (ACF) is a statistical tool used to study the correlation between a time series and its lagged values. In time series forecasting, the ACF is used to identify patterns and relationships between values in a time series at different lags, which can then be used to make predictions about future values.

        This function will implement the [`acf()`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html) function from the [`statsmodels`](https://www.statsmodels.org) library.

    ???+ abstract "Details"

        The acf at lag `0` (ie., `1`) is returned.

        For very long time series it is recommended to use `fft` convolution instead. When `fft` is `False` uses a simple, direct estimator of the autocovariances that only computes the first $nlag + 1$ values. This can be much faster when the time series is long and only a small number of autocovariances are needed.

        If `adjusted` is `True`, the denominator for the autocovariance is adjusted for the loss of data.

        The ACF measures the correlation between a time series and its lagged values at different lags. The correlation is calculated as the ratio of the covariance between the series and its lagged values to the product of their standard deviations. The ACF is typically plotted as a graph, with the lag on the `x`-axis and the correlation coefficient on the `y`-axis.

        The ACF at lag $k$ is defined as:

        $$
        ACF(k) = \frac{ Cov(Y_t, Y_{t-k}) } { Var(Y_t) \times Var(Y_{t-k}) }
        $$

        where:

        - $Y_t$ and $Y_{t-k}$ are the values of the time series at time $t$ and time $t-k$, respectively,
        - $Cov(Y_t, Y_{t-k})$ is the covariance between the two values, and
        - $Var(Y_t)$ and $Var(Y_{t-k})$ are the variances of the two values.

        ```
        ACF(k) = Cov(Y_t, Y_{t-k}) / (sqrt(Var(Y_t)) * sqrt(Var(Y_{t-k})))
        ```

        For a stationary series, this simplifies to:

        ```
        ACF(k) = Cov(Y_t, Y_{t-k}) / Var(Y_t)
        ```

        If the ACF shows a strong positive correlation at lag $k$, this means that values in the time series at time $t$ and time $t-k$ are strongly related. This can be useful in forecasting, as it suggests that past values can be used to predict future values. If the ACF shows a strong negative correlation at lag $k$, this means that values at time $t$ and time $t-k$ are strongly inversely related, which can also be useful in forecasting.

        The ACF can be used to identify the order of an autoregressive (AR) model, which is a type of model used in time series forecasting. The order of an AR model is the number of lags that are used to predict future values. The ACF can also be used to diagnose the presence of seasonality in a time series.

        Overall, the autocorrelation function is a valuable tool in time series forecasting, as it helps to identify patterns and relationships between values in a time series that can be used to make predictions about future values.

        The ACF can be calculated using the `acf()` function in the `statsmodels` package in Python. The function takes a time series array as input and returns an array of autocorrelation coefficients at different lags. The significance of the autocorrelation coefficients can be tested using the Ljung-Box test, which tests the null hypothesis that the autocorrelation coefficients are zero up to a certain lag. The Ljung-Box test can be performed using the `acorr_ljungbox()` function in the `statsmodels` package. If the p-value of the test is less than a certain significance level (e.g. $0.05$), then there is evidence of significant autocorrelation in the time series up to the specified lag.

    Params:
        x (ArrayLike):
            The time series data.
        adjusted (bool, optional):
            If `True`, then denominators for auto-covariance are $n-k$, otherwise $n$.<br>
            Defaults to `False`.
        nlags (Optional[int], optional):
            Number of lags to return autocorrelation for. If not provided, uses $\min(10 \times \text{log10}(nobs),nobs-1)$ (calculated with: `min(int(10 * np.log10(nobs)), nobs - 1)`). The returned value includes $lag 0$ (ie., $1$) so size of the acf vector is $(nlags + 1,)$.<br>
            Defaults to `None`.
        qstat (bool, optional):
            If `True`, also returns the Ljung-Box $q$ statistic and corresponding p-values for each autocorrelation coefficient; see the *Returns* section for details.<br>
            Defaults to `False`.
        fft (bool, optional):
            If `True`, computes the ACF via FFT.<br>
            Defaults to `True`.
        alpha (Optional[float], optional):
            If a number is given, the confidence intervals for the given level are returned. For instance if `alpha=0.05`, a $95\%$ confidence intervals are returned where the standard deviation is computed according to Bartlett"s formula.<br>
            Defaults to `None`.
        bartlett_confint (bool, optional):
            Confidence intervals for ACF values are generally placed at 2 standard errors around $r_k$. The formula used for standard error depends upon the situation. If the autocorrelations are being used to test for randomness of residuals as part of the ARIMA routine, the standard errors are determined assuming the residuals are white noise. The approximate formula for any lag is that standard error of each $r_k = \frac{1}{\sqrt{N}}$. See section 9.4 of [2] for more details on the $\frac{1}{\sqrt{N}}$ result. For more elementary discussion, see section 5.3.2 in [3]. For the ACF of raw data, the standard error at a lag $k$ is found as if the right model was an $\text{MA}(k-1)$. This allows the possible interpretation that if all autocorrelations past a certain lag are within the limits, the model might be an $\text{MA}$ of order defined by the last significant autocorrelation. In this case, a moving average model is assumed for the data and the standard errors for the confidence intervals should be generated using Bartlett's formula. For more details on Bartlett formula result, see section 7.2 in [2].<br>
            Defaults to `True`.
        missing (VALID_ACF_MISSING_OPTIONS, optional):
            A string in `["none", "raise", "conservative", "drop"]` specifying how the `NaN`'s are to be treated.

            - `"none"` performs no checks.
            - `"raise"` raises an exception if NaN values are found.
            - `"drop"` removes the missing observations and then estimates the autocovariances treating the non-missing as contiguous.
            - `"conservative"` computes the autocovariance using nan-ops so that nans are removed when computing the mean and cross-products that are used to estimate the autocovariance.

            When using `"conservative"`, $n$ is set to the number of non-missing observations.<br>
            Defaults to `"none"`.

    Returns:
        acf (np.ndarray):
            The autocorrelation function for lags `0, 1, ..., nlags`.<br>
            Shape `(nlags+1,)`.
        confint (Optional[np.ndarray]):
            Confidence intervals for the ACF at lags `0, 1, ..., nlags`.<br>
            Shape `(nlags + 1, 2)`.<br>
            Returned if `alpha` is not `None`.
        qstat (Optional[np.ndarray]):
            The Ljung-Box Q-Statistic for lags `1, 2, ..., nlags` (excludes lag zero).<br>
            Returned if `qstat` is `True`.
        pvalues (Optional[np.ndarray]):
            The p-values associated with the Q-statistics for lags `1, 2, ..., nlags` (excludes lag zero).<br>
            Returned if `qstat` is `True`.

    !!! success "Credit"
        - All credit goes to the [`statsmodels`](https://www.statsmodels.org/) library.

    !!! example "Examples"

        ```pycon {.py .python linenums="1" title="Test ACF without FFT"}
        >>> from pprint import pprint
        >>> from statsmodels.datasets import macrodata
        >>> from ts_stat_tests.algorithms.correlation import acf
        >>> data = macrodata.load_pandas()
        >>> x = data.data["realgdp"]
        >>> res_acf, res_confint, res_qstat, res_pvalues = acf(
        ...     x, nlags=40, qstat=True, alpha=0.05, fft=False
        ... )
        >>> pprint(res_acf[1:11])
        array([0.94804734, 0.87557484, 0.80668116, 0.75262542, 0.71376997,
               0.6817336 , 0.66290439, 0.65561048, 0.67094833, 0.70271992])
        >>> pprint(res_confint[1:11])
        array([[0.78471701, 1.11137767],
               [0.60238868, 1.14876099],
               [0.46677939, 1.14658292],
               [0.36500159, 1.14024925],
               [0.28894752, 1.13859242],
               [0.22604068, 1.13742653],
               [0.18077091, 1.14503787],
               [0.14974636, 1.16147461],
               [0.1429036 , 1.19899305],
               [0.15240228, 1.25303756]])
        >>> pprint(res_qstat[:10])
        array([132.14153858, 245.64616028, 342.67482586, 427.73868355,
               504.79657041, 575.6018536 , 643.03859337, 709.48449817,
               779.59123116, 857.06863862])
        >>> pprint(res_pvalues[:10])
        array([1.39323140e-030, 4.55631819e-054, 5.75108846e-074, 2.81773062e-091,
               7.36019524e-107, 4.26400770e-121, 1.30546283e-134, 6.49627091e-148,
               5.24937010e-162, 1.10078935e-177])
        ```

        ```pycon {.py .python linenums="1" title="Test ACF with FFT"}
        >>> from pprint import pprint
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.algorithms.correlation import acf
        >>> data = load_airline()
        >>> res_acf, res_confint, res_qstat, res_pvalues = acf(
        ...     data, nlags=40, qstat=True, alpha=0.05, fft=True
        ... )
        >>> pprint(res_acf[1:11])
        array([0.94804734, 0.87557484, 0.80668116, 0.75262542, 0.71376997,
               0.6817336 , 0.66290439, 0.65561048, 0.67094833, 0.70271992])
        >>> pprint(res_confint[1:11])
        array([[0.78471701, 1.11137767],
               [0.60238868, 1.14876099],
               [0.46677939, 1.14658292],
               [0.36500159, 1.14024925],
               [0.28894752, 1.13859242],
               [0.22604068, 1.13742653],
               [0.18077091, 1.14503787],
               [0.14974636, 1.16147461],
               [0.1429036 , 1.19899305],
               [0.15240228, 1.25303756]])
        >>> pprint(res_qstat[:10])
        array([132.14153858, 245.64616028, 342.67482586, 427.73868355,
               504.79657041, 575.6018536 , 643.03859337, 709.48449817,
               779.59123116, 857.06863862])
        >>> pprint(res_pvalues[:10])
        array([1.39323140e-030, 4.55631819e-054, 5.75108846e-074, 2.81773062e-091,
               7.36019524e-107, 4.26400770e-121, 1.30546283e-134, 6.49627091e-148,
               5.24937010e-162, 1.10078935e-177])
        ```

    ??? question "References"
        1. Parzen, E., 1963. On spectral analysis with missing observations and amplitude modulation. Sankhya: The Indian Journal of Statistics, Series A, pp.383-392.
        1. Brockwell and Davis, 1987. Time Series Theory and Methods.
        1. Brockwell and Davis, 2010. Introduction to Time Series and Forecasting, 2nd edition.

    ??? tip "See Also"
        - [`statsmodels.tsa.stattools.acf`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html): Estimate the autocorrelation function.
        - [`statsmodels.tsa.stattools.pacf`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html): Partial autocorrelation estimation.
        - [`statsmodels.tsa.stattools.ccf`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.ccf.html): The cross-correlation function.
        - [`ts_stat_tests.algorithms.correlation.acf`][ts_stat_tests.algorithms.correlation.acf]: Estimate the autocorrelation function
        - [`ts_stat_tests.algorithms.correlation.pacf`][ts_stat_tests.algorithms.correlation.pacf]: Partial autocorrelation estimate.
        - [`ts_stat_tests.algorithms.correlation.ccf`][ts_stat_tests.algorithms.correlation.ccf]: The cross-correlation function.
    """
    return st_acf(
        x=x,
        adjusted=adjusted,
        nlags=nlags,
        qstat=qstat,
        fft=fft,
        alpha=alpha,
        bartlett_confint=bartlett_confint,
        missing=missing,
    )


@overload
def pacf(
    x: ArrayLike1D,
    nlags: Optional[int] = None,
    method: VALID_PACF_METHOD_OPTIONS = "ywadjusted",
    *,
    alpha: None = None,
) -> np.ndarray: ...
@overload
def pacf(
    x: ArrayLike1D,
    nlags: Optional[int] = None,
    method: VALID_PACF_METHOD_OPTIONS = "ywadjusted",
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]: ...
@typechecked
def pacf(
    x: ArrayLike1D,
    nlags: Optional[int] = None,
    method: VALID_PACF_METHOD_OPTIONS = "ywadjusted",
    *,
    alpha: Optional[float] = None,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    r"""
    !!! note "Summary"

        The partial autocorrelation function (PACF) is a statistical tool used in time series forecasting to identify the direct relationship between two variables, controlling for the effect of the other variables in the time series. In other words, the PACF measures the correlation between a time series and its lagged values, while controlling for the effects of other intermediate lags.

        This function will implement the [`pacf()`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html) function from the [`statsmodels`](https://www.statsmodels.org) library.

    ???+ abstract "Details"

        Based on simulation evidence across a range of low-order ARMA models, the best methods based on root MSE are Yule-Walker (MLW), Levinson-Durbin (MLE) and Burg, respectively. The estimators with the lowest bias included these three in addition to OLS and OLS-adjusted. Yule-Walker (adjusted) and Levinson-Durbin (adjusted) performed consistently worse than the other options.

        The PACF is a plot of the correlation between a time series and its lagged values, controlling for the effect of other lags. The PACF is useful for identifying the order of an autoregressive (AR) model, which is a type of model used in time series forecasting. The order of an AR model is the number of lags that are used to predict future values.

        The PACF at lag $k$ is defined as:

        $$
        PACF(k) = Corr \left( Y_t, Y_{t-k} \mid Y_{t-1}, Y_{t-2}, ..., Y_{t-k+1} \right)
        $$

        where:

        - $Y_t$ and $Y_{t-k}$ are the values of the time series at time $t$ and time $t-k$, respectively, and
        - $Y_{t-1}, Y_{t-2}, ..., Y_{t-k+1}$ are the values of the time series at intervening lags.
        - $Corr()$ denotes the correlation coefficient between two variables.

        ```
        PACF(k) = Corr(Y_t, Y_{t-k} | Y_{t-1}, Y_{t-2}, ..., Y_{t-k+1})
        ```

        The PACF is calculated using the Yule-Walker equations, which are a set of linear equations that describe the relationship between a time series and its lagged values. The PACF is calculated as the difference between the correlation coefficient at lag $k$ and the correlation coefficient at lag $k-1$, controlling for the effects of intermediate lags.

        The PACF is typically plotted as a graph, with the lag on the `x`-axis and the correlation coefficient on the `y`-axis. If the PACF shows a strong positive correlation at lag $k$, this means that values in the time series at time $t$ and time $t-k$ are strongly related, after controlling for the effects of intermediate lags. This suggests that past values can be used to predict future values using an AR model with an order of $k$.

        Overall, the partial autocorrelation function is a valuable tool in time series forecasting, as it helps to identify the order of an autoregressive model and to control for the effects of intermediate lags. By identifying the direct relationship between two variables, the PACF can help to improve the accuracy of time series forecasting models.

        The PACF can be calculated using the pacf() function in the statsmodels package in Python. The function takes a time series array as input and returns an array of partial autocorrelation coefficients at different lags. The significance of the partial autocorrelation coefficients can be tested using the same Ljung-Box test as for the ACF. If the p-value of the test is less than a certain significance level (e.g. $0.05$), then there is evidence of significant partial autocorrelation in the time series up to the specified lag.

    Params:
        x (ArrayLike1D):
            Observations of time series for which pacf is calculated.
        nlags (Optional[int], optional):
            Number of lags to return autocorrelation for. If not provided, uses $min(10 \times log10(nobs) , (\frac{nobs}{2}-1))$ (calculated with: `min(int(10*np.log10(nobs)), nobs // 2 - 1)`). The returned value includes lag `0` (ie., `1`) so size of the pacf vector is $(nlags + 1,)$.<br>
            Defaults to `None`.
        method (VALID_PACF_METHOD_OPTIONS, optional):
            Specifies which method for the calculations to use.

            - `"yw"` or `"ywadjusted"`: Yule-Walker with sample-size adjustment in denominator for acovf. Default.
            - `"ywm"` or `"ywmle"`: Yule-Walker without adjustment.
            - `"ols"`: regression of time series on lags of it and on constant.
            - `"ols-inefficient"`: regression of time series on lags using a single common sample to estimate all pacf coefficients.
            - `"ols-adjusted"`: regression of time series on lags with a bias adjustment.
            - `"ld"` or `"ldadjusted"`: Levinson-Durbin recursion with bias correction.
            - `"ldb"` or `"ldbiased"`: Levinson-Durbin recursion without bias correction.<br>

            Defaults to `"ywadjusted"`.
        alpha (Optional[float], optional):
            If a number is given, the confidence intervals for the given level are returned. For instance if `alpha=.05`, $95\%$ confidence intervals are returned where the standard deviation is computed according to $\frac{1}{\sqrt{len(x)}}$.<br>
            Defaults to `None`.

    Returns:
        pacf (np.ndarray):
            The partial autocorrelations for lags `0, 1, ..., nlags`.<br>
            Shape `(nlags+1,)`.
        confint (Optional[np.ndarray]):
            Confidence intervals for the PACF at lags `0, 1, ..., nlags`.<br>
            Shape `(nlags + 1, 2)`.<br>
            Returned if `alpha` is not `None`.

    !!! success "Credit"
        - All credit goes to the [`statsmodels`](https://www.statsmodels.org/) library.

    !!! example "Examples"

        ```pycon {.py .python linenums="1" title="Test PACF using Yule-Walker method with sample-size adjustment"}
        >>> from pprint import pprint
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.algorithms.correlation import pacf
        >>> data = load_airline()
        >>> res_acf, res_confint = pacf(data, nlags=40, alpha=0.05)
        >>> pprint(res_acf[1:11])
        array([ 0.95467704, -0.26527732,  0.05546955,  0.10885622,  0.08112579,
                0.00412541,  0.15616955,  0.10370833,  0.28878144,  0.20691805])
        >>> pprint(res_confint[1:11])
        array([[ 0.79134671,  1.11800737],
               [-0.42860765, -0.10194698],
               [-0.10786078,  0.21879988],
               [-0.05447412,  0.27218655],
               [-0.08220455,  0.24445612],
               [-0.15920493,  0.16745574],
               [-0.00716078,  0.31949988],
               [-0.059622  ,  0.26703866],
               [ 0.12545111,  0.45211177],
               [ 0.04358772,  0.37024838]])
        ```

        ```pycon {.py .python linenums="1" title="Test PACF using Yule-Walker method without adjustment"}
        >>> from pprint import pprint
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.algorithms.correlation import pacf
        >>> data = load_airline()
        >>> res_acf, res_confint = pacf(data, nlags=40, method="ywm", alpha=0.05)
        >>> pprint(res_acf[1:11])
        array([ 0.94804734, -0.22942187,  0.03814778,  0.09378544,  0.0736067 ,
                0.0077276 ,  0.12559713,  0.08995134,  0.23248854,  0.16605126])
        >>> pprint(res_confint[1:11])
        array([[ 0.78471701,  1.11137767],
               [-0.39275221, -0.06609154],
               [-0.12518255,  0.20147811],
               [-0.06954489,  0.25711577],
               [-0.08972363,  0.23693703],
               [-0.15560273,  0.17105793],
               [-0.0377332 ,  0.28892746],
               [-0.07337899,  0.25328168],
               [ 0.06915821,  0.39581887],
               [ 0.00272093,  0.32938159]])
        ```

        ```pycon {.py .python linenums="1" title="Test PACF using regression of time series"}
        >>> from pprint import pprint
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.algorithms.correlation import pacf
        >>> data = load_airline()
        >>> res_acf, res_confint = pacf(data, nlags=40, method="ols", alpha=0.05)
        >>> pprint(res_acf[1:11])
        array([ 0.95893198, -0.32983096,  0.2018249 ,  0.14500798,  0.25848232,
               -0.02690283,  0.20433019,  0.15607896,  0.56860841,  0.29256358])
        >>> pprint(res_confint[1:11])
        array([[ 0.79560165,  1.12226231],
               [-0.49316129, -0.16650062],
               [ 0.03849457,  0.36515523],
               [-0.01832235,  0.30833831],
               [ 0.09515198,  0.42181265],
               [-0.19023316,  0.1364275 ],
               [ 0.04099986,  0.36766053],
               [-0.00725137,  0.31940929],
               [ 0.40527808,  0.73193874],
               [ 0.12923325,  0.45589391]])
        ```

        ```pycon {.py .python linenums="1" title="Test PACF using regression of time series on lags with inefficient optimisation"}
        >>> from pprint import pprint
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.algorithms.correlation import pacf
        >>> data = load_airline()
        >>> res_acf, res_confint = pacf(data, nlags=40, method="ols-inefficient", alpha=0.05)
        >>> pprint(res_acf[1:11])
        array([ 0.94692978, -0.3540491 ,  0.18292698,  0.12813227,  0.23647898,
               -0.04596983,  0.19748537,  0.12877966,  0.55357665,  0.22081591])
        >>> pprint(res_confint[1:11])
        array([[ 0.78359944,  1.11026011],
               [-0.51737943, -0.19071876],
               [ 0.01959665,  0.34625731],
               [-0.03519806,  0.2914626 ],
               [ 0.07314865,  0.39980932],
               [-0.20930016,  0.1173605 ],
               [ 0.03415504,  0.3608157 ],
               [-0.03455067,  0.29211   ],
               [ 0.39024632,  0.71690698],
               [ 0.05748558,  0.38414625]])
        ```

        ```pycon {.py .python linenums="1" title="Test PACF using regression of time series on lags with a bias adjustment"}
        >>> from pprint import pprint
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.algorithms.correlation import pacf
        >>> data = load_airline()
        >>> res_acf, res_confint = pacf(data, nlags=40, method="ols-adjusted", alpha=0.05)
        >>> pprint(res_acf[1:11])
        array([ 0.9656378 , -0.33447646,  0.20611905,  0.14915107,  0.26778024,
               -0.02807252,  0.21477042,  0.16526008,  0.60651564,  0.31439668])
        >>> pprint(res_confint[1:11])
        array([[ 0.80230746,  1.12896813],
               [-0.49780679, -0.17114613],
               [ 0.04278872,  0.36944938],
               [-0.01417926,  0.3124814 ],
               [ 0.10444991,  0.43111057],
               [-0.19140285,  0.13525782],
               [ 0.05144009,  0.37810076],
               [ 0.00192974,  0.32859041],
               [ 0.4431853 ,  0.76984597],
               [ 0.15106635,  0.47772701]])
        ```

        ```pycon {.py .python linenums="1" title="Test PACF using Levinson-Durbin recursion with bias correction"}
        >>> from pprint import pprint
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.algorithms.correlation import pacf
        >>> data = load_airline()
        >>> res_acf, res_confint = pacf(data, nlags=40, method="ldadjusted", alpha=0.05)
        >>> pprint(res_acf[1:11])
        array([ 0.95467704, -0.26527732,  0.05546955,  0.10885622,  0.08112579,
                0.00412541,  0.15616955,  0.10370833,  0.28878144,  0.20691805])
        >>> pprint(res_confint[1:11])
        array([[ 0.79134671,  1.11800737],
               [-0.42860765, -0.10194698],
               [-0.10786078,  0.21879988],
               [-0.05447412,  0.27218655],
               [-0.08220455,  0.24445612],
               [-0.15920493,  0.16745574],
               [-0.00716078,  0.31949988],
               [-0.059622  ,  0.26703866],
               [ 0.12545111,  0.45211177],
               [ 0.04358772,  0.37024838]])
        ```

        ```pycon {.py .python linenums="1" title="Test PACF using Levinson-Durbin recursion without bias correction"}
        >>> from pprint import pprint
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.algorithms.correlation import pacf
        >>> data = load_airline()
        >>> res_acf, res_confint = pacf(data, nlags=40, method="ldbiased", alpha=0.05)
        >>> pprint(res_acf[1:11])
        array([ 0.94804734, -0.22942187,  0.03814778,  0.09378544,  0.0736067 ,
                0.0077276 ,  0.12559713,  0.08995134,  0.23248854,  0.16605126])
        >>> pprint(res_confint[1:11])
        array([[ 0.78471701,  1.11137767],
               [-0.39275221, -0.06609154],
               [-0.12518255,  0.20147811],
               [-0.06954489,  0.25711577],
               [-0.08972363,  0.23693703],
               [-0.15560273,  0.17105793],
               [-0.0377332 ,  0.28892746],
               [-0.07337899,  0.25328168],
               [ 0.06915821,  0.39581887],
               [ 0.00272093,  0.32938159]])
        ```

    ??? question "References"
        1. Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: forecasting and control. John Wiley & Sons, p. 66.
        1. Brockwell, P.J. and Davis, R.A., 2016. Introduction to time series and forecasting. Springer.

    ??? tip "See Also"
        - [`statsmodels.tsa.stattools.acf`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html): Estimate the autocorrelation function.
        - [`statsmodels.tsa.stattools.pacf`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html): Partial autocorrelation estimation.
        - [`statsmodels.tsa.stattools.ccf`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.ccf.html): The cross-correlation function.
        - [`statsmodels.tsa.stattools.pacf_yw`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf_yw.html): Partial autocorrelation estimation using Yule-Walker.
        - [`statsmodels.tsa.stattools.pacf_ols`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf_ols.html): Partial autocorrelation estimation using OLS.
        - [`statsmodels.tsa.stattools.pacf_burg`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf_burg.html): Partial autocorrelation estimation using Burg's method.
        - [`ts_stat_tests.algorithms.correlation.acf`][ts_stat_tests.algorithms.correlation.acf]: Estimate the autocorrelation function
        - [`ts_stat_tests.algorithms.correlation.pacf`][ts_stat_tests.algorithms.correlation.pacf]: Partial autocorrelation estimate.
        - [`ts_stat_tests.algorithms.correlation.ccf`][ts_stat_tests.algorithms.correlation.ccf]: The cross-correlation function.
    """
    return st_pacf(
        x=x,
        nlags=nlags,
        method=method,
        alpha=alpha,
    )


@overload
def ccf(
    x: ArrayLike,
    y: ArrayLike,
    adjusted: bool = True,
    fft: bool = True,
    *,
    nlags: Optional[int] = None,
    alpha: None = None,
) -> np.ndarray: ...
@overload
def ccf(
    x: ArrayLike,
    y: ArrayLike,
    adjusted: bool = True,
    fft: bool = True,
    *,
    nlags: Optional[int] = None,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]: ...
@typechecked
def ccf(
    x: ArrayLike,
    y: ArrayLike,
    adjusted: bool = True,
    fft: bool = True,
    *,
    nlags: Optional[int] = None,
    alpha: Optional[float] = None,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    !!! note "Summary"

        The cross-correlation function (CCF) is a statistical tool used in time series forecasting to measure the correlation between two time series at different lags. It is used to study the relationship between two time series, and can help to identify lead-lag relationships and causal effects.

        This function will implement the [`ccf()`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.ccf.html) function from the [`statsmodels`](https://www.statsmodels.org) library.

    ???+ abstract "Details"

        If `adjusted` is `True`, the denominator for the autocovariance is adjusted.

        The CCF measures the correlation between two time series at different lags. It is calculated as the ratio of the covariance between the two series at lag k to the product of their standard deviations. The CCF is typically plotted as a graph, with the lag on the `x`-axis and the correlation coefficient on the `y`-axis.

        The CCF at lag k is defined as:

        $$
        CCF(k) = Corr(X_t, Y_{t-k})
        $$

        where:

        - $X_t$ and $Y_{t-k}$ are the values of the two time series at time $t$ and time $t-k$, respectively.
        - $Corr()$ denotes the correlation coefficient between two variables.

        ```
        CCF(k) = Corr(X_t, Y_{t-k})
        ```

        If the CCF shows a strong positive correlation at lag $k$, this means that changes in one time series at time $t$ are strongly related to changes in the other time series at time $t-k$. This suggests a lead-lag relationship between the two time series, where changes in one series lead changes in the other series by a certain number of periods. The CCF can be used to estimate the time lag between the two time series.

        The CCF can also help to identify causal relationships between two time series. If the CCF shows a strong positive correlation at lag $k$, and the lag is consistent with a causal relationship between the two time series, this suggests that changes in one time series are causing changes in the other time series.

        Overall, the cross-correlation function is a valuable tool in time series forecasting, as it helps to study the relationship between two time series and to identify lead-lag relationships and causal effects. By identifying the relationship between two time series, the CCF can help to improve the accuracy of time series forecasting models.

        The CCF can be calculated using the `ccf()` function in the `statsmodels` package in Python. The function takes two time series arrays as input and returns an array of cross-correlation coefficients at different lags. The significance of the cross-correlation coefficients can be tested using a similar test to the Ljung-Box test, such as the Box-Pierce test or the Breusch-Godfrey test. These tests can be performed using the `boxpierce()` and `lm()` functions in the `statsmodels` package, respectively. If the p-value of the test is less than a certain significance level (e.g. $0.05$), then there is evidence of significant cross-correlation between the two time series at the specified lag.

    Params:
        x (ArrayLike):
            The time series data to use in the calculation.
        y (ArrayLike):
            The time series data to use in the calculation.
        adjusted (bool, optional):
            If `True`, then denominators for cross-correlation is $n-k$, otherwise $n$.<br>
            Defaults to `True`.
        fft (bool, optional):
            If `True`, use FFT convolution. This method should be preferred for long time series.<br>
            Defaults to `True`.
        nlags (Optional[int], optional):
            Number of lags to return cross-correlations for. If not provided, the number of lags equals len(x).
            Defaults to `None`.
        alpha (Optional[float], optional):
            If a number is given, the confidence intervals for the given level are returned. For instance if `alpha=.05`, 95% confidence intervals are returned where the standard deviation is computed according to `1/sqrt(len(x))`.
            Defaults to `None`.

    Returns:
        ccf (np.ndarray):
            The cross-correlation function of `x` and `y`.

    !!! success "Credit"
        - All credit goes to the [`statsmodels`](https://www.statsmodels.org/) library.

    !!! example "Examples"

        ```pycon {.py .python linenums="1" title="Test CCF without FFT"}
        >>> from pprint import pprint
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.algorithms.correlation import ccf
        >>> data = load_airline()
        >>> res_ccf = ccf(data, data + 1, adjusted=True, fft=False)
        >>> pprint(res_ccf[1:11])
        array([0.95467704, 0.88790688, 0.82384458, 0.774129  , 0.73944515,
               0.71137419, 0.69677541, 0.69417581, 0.71567822, 0.75516171])
        ```

        ```pycon {.py .python linenums="1" title="Test CCF with FFT"}
        >>> from pprint import pprint
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.algorithms.correlation import ccf
        >>> data = load_airline()
        >>> res_ccf = ccf(data, data + 1, adjusted=True, fft=True)
        >>> pprint(res_ccf[1:11])
        array([0.95467704, 0.88790688, 0.82384458, 0.774129  , 0.73944515,
               0.71137419, 0.69677541, 0.69417581, 0.71567822, 0.75516171])
        ```

    ??? tip "See Also"
        - [`statsmodels.tsa.stattools.acf`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html): Estimate the autocorrelation function.
        - [`statsmodels.tsa.stattools.pacf`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html): Partial autocorrelation estimation.
        - [`statsmodels.tsa.stattools.ccf`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.ccf.html): The cross-correlation function.
        - [`ts_stat_tests.algorithms.correlation.acf`][ts_stat_tests.algorithms.correlation.acf]: Estimate the autocorrelation function
        - [`ts_stat_tests.algorithms.correlation.pacf`][ts_stat_tests.algorithms.correlation.pacf]: Partial autocorrelation estimate.
        - [`ts_stat_tests.algorithms.correlation.ccf`][ts_stat_tests.algorithms.correlation.ccf]: The cross-correlation function.
    """
    return st_ccf(
        x=x,
        y=y,
        adjusted=adjusted,
        fft=fft,
        nlags=nlags,
        alpha=alpha,
    )


@typechecked
def lb(
    x: ArrayLike,
    lags: Optional[Union[int, ArrayLike]] = None,
    boxpierce: bool = False,
    model_df: int = 0,
    period: Optional[int] = None,
    return_df: bool = True,
    auto_lag: bool = False,
) -> pd.DataFrame:
    r"""
    !!! note "Summary"

        The Ljung-Box test is a statistical test used in time series forecasting to test for the presence of autocorrelation in the residuals of a model. The test is based on the autocorrelation function (ACF) of the residuals, and can be used to assess the adequacy of a time series model and to identify areas for improvement.

        This function will implement the [`acorr_ljungbox()`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html) function from the [`statsmodels`](https://www.statsmodels.org) library.

    ???+ abstract "Details"

        The Ljung-Box and Box-Pierce statistics differ in how they scale the autocorrelation function; the Ljung-Box test has better finite-sample properties.

        The test statistic is calculated as:

        $$
        Q(m) = n(n+2) \times \sum_{k=1}^m \left( \frac{ r_k^2 }{ n-k } \right)
        $$

        where:

        - $n$ is the sample size,
        - $m$ is the maximum lag being tested,
        - $r_k$ is the sample autocorrelation at lag $k$, and
        - $\sum$ ($sum$) denotes the sum over $k$ from $1$ to $m$.

        ```
        Q(m) = n(n+2) * Sum(r_k^2 / (n-k))
        ```

        Under the null hypothesis, the test statistic follows a chi-squared distribution with degrees of freedom equal to $m-p$, where $p$ is the number of parameters estimated in fitting the time series model.

        The Ljung-Box test is performed by calculating the autocorrelation function (ACF) of the residuals from a time series model, and then comparing the ACF values to the expected values under the null hypothesis of no autocorrelation. The test statistic is calculated as the sum of the squared autocorrelations up to a given lag, and is compared to a chi-squared distribution with degrees of freedom equal to the number of lags tested.

        If the test statistic is greater than the critical value from the chi-squared distribution, then the null hypothesis of no autocorrelation is rejected, indicating that there is evidence of autocorrelation in the residuals. This suggests that the time series model is inadequate, and that additional terms may need to be added to the model to account for the remaining autocorrelation.

        If the test statistic is less than the critical value from the chi-squared distribution, then the null hypothesis of no autocorrelation is not rejected, indicating that there is no evidence of autocorrelation in the residuals. This suggests that the time series model is adequate, and that no further improvements are needed.

        Overall, the Ljung-Box test is a valuable tool in time series forecasting, as it helps to assess the adequacy of a time series model and to identify areas for improvement. By testing for autocorrelation in the residuals, the test helps to ensure that the model is accurately capturing the underlying patterns in the time series data.

        The Ljung-Box test can be calculated using the `acorr_ljungbox()` function in the `statsmodels` package in Python. The function takes a time series array and the maximum lag $m$ as input, and returns an array of Q-statistics and associated p-values for each lag up to $m$. If the p-value of the test is less than a certain significance level (e.g. $0.05$), then there is evidence of significant autocorrelation in the time series up to the specified lag.

    Params:
        x (ArrayLike):
            The data series. The data is demeaned before the test statistic is computed.
        lags (Optional[Union[int, ArrayLike]], optional):
            If lags is an integer (`int`) then this is taken to be the largest lag that is included, the test result is reported for all smaller lag length. If lags is a list or array, then all lags are included up to the largest lag in the list, however only the tests for the lags in the list are reported. If lags is `None`, then the default maxlag is currently $\min(\frac{nobs}{2}-2,40)$ (calculated with: `min(nobs // 2 - 2, 40)`). The default number of `lags` changes if `period` is set.<br>
            !!! deprecation "Deprecation"
                After `statsmodels` version `0.12`, this will calculation change from

                $$
                \min(\frac{nobs}{2}-2,40)
                $$

                to

                $$
                \min(10,\frac{nobs}{5})
                $$
            Defaults to `None`.
        boxpierce (bool, optional):
            If `True`, then additional to the results of the Ljung-Box test also the Box-Pierce test results are returned.<br>
            Defaults to `False`.
        model_df (int, optional):
            Number of degrees of freedom consumed by the model. In an ARMA model, this value is usually $p+q$ where $p$ is the AR order and $q$ is the MA order. This value is subtracted from the degrees-of-freedom used in the test so that the adjusted dof for the statistics are $lags - model_df$. If $lags - model_df <= 0$, then `NaN` is returned.<br>
            Defaults to `0`.
        period (Optional[int], optional):
            The period of a Seasonal time series. Used to compute the max lag for seasonal data which uses $\min(2 \times period, \frac{nobs}{5})$ (calculated with: `min(2*period,nobs//5)`) if set. If `None`, then the default rule is used to set the number of lags. When set, must be $>= 2$.<br>
            Defaults to `None`.
        return_df (bool, optional):
            Flag indicating whether to return the result as a single DataFrame with columns `lb_stat`, `lb_pvalue`, and optionally `bp_stat` and `bp_pvalue`. Set to `True` to return the DataFrame or `False` to continue returning the $2-4$ output. If `None` (the default), a warning is raised.
            !!! deprecation "Deprecation"
                After `statsmodels` version `0.12`, this will become the only return method.
            Defaults to `True`.
        auto_lag (bool, optional):
            Flag indicating whether to automatically determine the optimal lag length based on threshold of maximum correlation value.<br>
            Defaults to `False`.

    Returns:
        lbvalue (Union[float, np.ndarray]):
            The Ljung-Box test statistic.
        pvalue (Union[float, np.ndarray]):
            The p-value based on chi-square distribution. The p-value is computed as $1-\text{cdf}(lbvalue,dof)$ where $dof$ is $lag - model\_df$ (calculated with: `1.0 - chi2.cdf(lbvalue, dof)`). If $lag - model\_df <= 0$, then `NaN` is returned for the `pvalue`.
        bpvalue (Optional[Union[float, np.ndarray]]):
            The test statistic for Box-Pierce test.
        bppvalue (Optional[Union[float, np.ndarray]]):
            The p-value based for Box-Pierce test on chi-square distribution. The p-value is computed as $1-\text{cdf}(bpvalue,dof)$ where $dof$ is $lag - model_df$ (calculated with: `1.0 - chi2.cdf(bpvalue, dof)`). If $lag - model_df <= 0$, then `NaN` is returned for the `pvalue`.

    !!! success "Credit"
        - All credit goes to the [`statsmodels`](https://www.statsmodels.org/) library.

    !!! example "Examples"

        ```pycon {.py .python linenums="1" title="Python"}
        >>> import statsmodels.api as sm
        >>> from ts_stat_tests.algorithms.correlation import lb
        >>> data = sm.datasets.sunspots.load_pandas().data
        >>> res = sm.tsa.ARIMA(data["SUNACTIVITY"], order=(1, 0, 1)).fit()
        >>> lb(res.resid, lags=[10], return_df=True)
            lb_stat     lb_pvalue
        10  214.106992  1.827374e-40
        ```

    ??? question "References"
        - Green, W. "Econometric Analysis," 5th ed., Pearson, 2003.
        - J. Carlos Escanciano, Ignacio N. Lobato "An automatic Portmanteau test for serial correlation"., Volume 151, 2009.

    ??? tip "See Also"
        - [`statsmodels.regression.linear_model.OLS.fit`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.fit.html):
        - [`statsmodels.regression.linear_model.RegressionResults`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html): The output results of a linear regression model.
        - [`statsmodels.stats.diagnostic.acorr_ljungbox`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html): Ljung-Box test for serial correlation.
        - [`statsmodels.stats.diagnostic.acorr_lm`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_lm.html): Lagrange Multiplier tests for autocorrelation.
        - [`statsmodels.stats.diagnostic.acorr_breusch_godfrey`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_breusch_godfrey.html): Breusch-Godfrey test for serial correlation.
        - [`ts_stat_tests.algorithms.correlation.lb`][ts_stat_tests.algorithms.correlation.lb]: Ljung-Box test of autocorrelation in residuals.
        - [`ts_stat_tests.algorithms.correlation.lm`][ts_stat_tests.algorithms.correlation.lm]: Lagrange Multiplier tests for autocorrelation.
        - [`ts_stat_tests.algorithms.correlation.bglm`][ts_stat_tests.algorithms.correlation.bglm]: Breusch-Godfrey Lagrange Multiplier tests for residual autocorrelation.
    """
    return acorr_ljungbox(
        x=x,
        lags=lags,
        boxpierce=boxpierce,
        model_df=model_df,
        period=period,
        return_df=return_df,
        auto_lag=auto_lag,
    )


@overload
def lm(
    resid: ArrayLike,
    nlags: Optional[int] = None,
    *,
    store: Literal[False] = False,
    period: Optional[int] = None,
    ddof: int = 0,
    cov_type: Literal["nonrobust"] = "nonrobust",
    cov_kwargs: Optional[dict] = None,
) -> tuple[float, float, float, float]: ...
@overload
def lm(
    resid: ArrayLike,
    nlags: Optional[int] = None,
    *,
    store: Literal[True],
    period: Optional[int] = None,
    ddof: int = 0,
    cov_type: Literal["nonrobust"] = "nonrobust",
    cov_kwargs: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, float, float, ResultsStore]: ...
@overload
def lm(
    resid: ArrayLike,
    nlags: Optional[int] = None,
    *,
    store: Literal[False] = False,
    period: Optional[int] = None,
    ddof: int = 0,
    cov_type: VALID_LM_COV_TYPE_OPTIONS,
    cov_kwargs: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, float, float]: ...
@overload
def lm(
    resid: ArrayLike,
    nlags: Optional[int] = None,
    *,
    store: Literal[True],
    period: Optional[int] = None,
    ddof: int = 0,
    cov_type: VALID_LM_COV_TYPE_OPTIONS,
    cov_kwargs: Optional[dict] = None,
) -> tuple[float, float, float, float, ResultsStore]: ...
@typechecked
def lm(
    resid: ArrayLike,
    nlags: Optional[int] = None,
    *,
    store: bool = False,
    period: Optional[int] = None,
    ddof: int = 0,
    cov_type: Union[Literal["nonrobust"], VALID_LM_COV_TYPE_OPTIONS] = "nonrobust",
    cov_kwargs: Optional[dict] = None,
) -> Union[
    tuple[np.ndarray, np.ndarray, float, float],
    tuple[np.ndarray, np.ndarray, float, float, ResultsStore],
    tuple[float, float, float, float],
    tuple[float, float, float, float, ResultsStore],
]:
    """
    !!! note "Summary"

        The Lagrange Multiplier (LM) test is a statistical test used in time series forecasting to test for the presence of autocorrelation in a model. The test is based on the residual sum of squares (RSS) of a time series model, and can be used to assess the adequacy of the model and to identify areas for improvement.

        This function will implement the [`acorr_lm()`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_lm.html) function from the [`statsmodels`](https://www.statsmodels.org) library.

    ???+ abstract "Details"

        This is a generic Lagrange Multiplier (LM) test for autocorrelation. It returns Engle's ARCH test if `resid` is the squared residual array. The Breusch-Godfrey test is a variation on this LM test with additional exogenous variables in the auxiliary regression.

        The LM test statistic is computed as

        $$
        LM = (n_{obs} - ddof) \\times R^2,
        $$

        ```
        LM = (n_obs - ddof) * R^2
        ```

        where $R^2$ is the coefficient of determination from the **auxiliary regression** of the residuals on their own `nlags` lags (and any additional regressors included in the model), $n_{obs}$ is the number of observations, and $ddof$ is the model degrees of freedom lost due to parameter estimation.

        <!-- Previous algorithm included below

        $$
        LM = n \\times (n+2) \\times \\sum_{k=1}^m \\left( \\frac { r_k^2 }{ n-k } \\right) - 2 \\times (n-1) \\times (n-2) \\times \\sum_{k=1}^m \\left( r_k \\times \\frac { r_{k+1} }{ n-k } \\right)
        $$

        where:

        - $n$ is the sample size,
        - $m$ is the maximum lag being tested,
        - $r_k$ is the sample autocorrelation at lag $k$, and
        - $\\sum$ ($sum$) denotes the sum over $k$ from $1$ to $m$.

        ```
        LM = n * (n+2) * Sum(r_k^2 / (n-k)) - 2 * (n-1) * (n-2) * Sum(r_k * r_(k+1) / (n-k))
        ```

        -->

        In practice, the LM test proceeds by:

        - Fitting a time series model to the data and obtaining the residuals.
        - Running an auxiliary regression of these residuals on their past `nlags` values (and any relevant exogenous variables).
        - Computing the LM statistic as $(n_{obs} - ddof) \\times R^2$ from this auxiliary regression.

        Under the null hypothesis that the autocorrelations up to the specified lag are zero (no serial correlation in the residuals), the LM statistic is asymptotically distributed as a chi-squared random variable with degrees of freedom equal to the number of lagged residual terms included in the auxiliary regression (i.e. the number of lags being tested, adjusted for any restrictions implied by the model).

        If the test statistic is greater than the critical value from the chi-squared distribution (or equivalently, if the p-value is less than a chosen significance level such as $0.05$), then the null hypothesis of no autocorrelation is rejected, indicating that there is evidence of autocorrelation in the residuals. This suggests that the time series model may be inadequate and that additional terms may need to be added to the model to account for the remaining autocorrelation.

        If the test statistic is less than the critical value from the chi-squared distribution, then the null hypothesis of no autocorrelation is not rejected, indicating that there is no evidence of autocorrelation in the residuals. This suggests that the time series model is adequate and that no further improvements are needed with respect to serial correlation.

        The LM test is a generalization of the Durbin-Watson test, which is a simpler test that only tests for first-order autocorrelation. The LM test can be used to test for higher-order autocorrelation and is more powerful than the Durbin-Watson test.

        Overall, the Lagrange Multiplier test is a valuable tool in time series forecasting, as it helps to assess the adequacy of a time series model and to identify areas for improvement. By testing for autocorrelation in the residuals, the test helps to ensure that the model is accurately capturing the underlying patterns in the time series data.

        The LM test can be calculated using the `acorr_lm()` function in the `statsmodels` package in Python. The function takes a time series array and the maximum lag `m` as input, and returns the LM test statistic and associated p-value. If the p-value of the test is less than a certain significance level (e.g. $0.05$), then there is evidence of significant autocorrelation in the time series up to the specified lag.

    Params:
        resid (ArrayLike):
            Time series to test.
        nlags (Optional[int], optional):
            Highest lag to use.<br>
            Defaults to `None`.
            !!! deprecation "Deprecation"
                The behavior of this parameter will change after `statsmodels` version `0.12`.
        store (bool, optional):
            If `True` then the intermediate results are also returned.<br>
            Defaults to `False`.
        period (Optional[int], optional):
            The period of a Seasonal time series. Used to compute the max lag for seasonal data which uses $\\min(2 \\times period, \\frac{nobs}{5})$ (calculated with: `min(2*period,nobs//5)`) if set. If `None`, then the default rule is used to set the number of lags. When set, must be $>=$ `2`.<br>
            Defaults to `None`.
        ddof (int, optional):
            The number of degrees of freedom consumed by the model used to produce resid<br>
            Defaults to `0`.
        cov_type (Union[Literal["nonrobust"], VALID_LM_COV_TYPE_OPTIONS], optional):
            Covariance type. The default is `"nonrobust"` which uses the classic OLS covariance estimator. Specify one of `"HC0"`, `"HC1"`, `"HC2"`, `"HC3"` to use White's covariance estimator. All covariance types supported by `OLS.fit` are accepted.<br>
            Defaults to `"nonrobust"`.
        cov_kwargs (Optional[dict], optional):
            Dictionary of covariance options passed to `OLS.fit`. See [`OLS.fit`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.fit.html) for more details.<br>
            Defaults to `None`.

    Returns:
        lm (float):
            Lagrange multiplier test statistic.
        lmpval (float):
            The `p-value` for Lagrange multiplier test.
        fval (float):
            The `f-statistic` of the F test, alternative version of the same test based on F test for the parameter restriction.
        fpval (float):
            The `p-value` of the F test.
        res_store (Optional[ResultsStore]):
            Intermediate results. Only returned if `store=True`.

    !!! success "Credit"
        - All credit goes to the [`statsmodels`](https://www.statsmodels.org/) library.

    !!! example "Examples"

        ```pycon {.py .python linenums="1" title="Test Lagrange Multiplier for autocorrelation"}
        >>> from pprint import pprint
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.algorithms.correlation import lm
        >>> data = load_airline()
        >>> res_lm, res_lmpval, res_fval, res_fpval = lm(data)
        >>> pprint(res_lm)
        128.09655717844828
        >>> pprint(res_lmpval)
        1.1416848684314836e-22
        >>> pprint(res_fval)
        266.89301496118736
        >>> pprint(res_fpval)
        2.36205831339912e-78
        ```

    ??? tip "See Also"
        - [`statsmodels.regression.linear_model.OLS.fit`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.fit.html): Fit a linear model.
        - [`statsmodels.regression.linear_model.RegressionResults`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html): The output results of a linear regression model.
        - [`statsmodels.stats.diagnostic.het_arch`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_arch.html#statsmodels.stats.diagnostic.het_arch): Conditional heteroskedasticity testing.
        - [`statsmodels.stats.diagnostic.acorr_ljungbox`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html): Ljung-Box test for serial correlation.
        - [`statsmodels.stats.diagnostic.acorr_lm`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_lm.html): Lagrange Multiplier tests for autocorrelation.
        - [`statsmodels.stats.diagnostic.acorr_breusch_godfrey`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_breusch_godfrey.html): Breusch-Godfrey test for serial correlation.
        - [`ts_stat_tests.algorithms.correlation.lb`][ts_stat_tests.algorithms.correlation.lb]: Ljung-Box test of autocorrelation in residuals.
        - [`ts_stat_tests.algorithms.correlation.lm`][ts_stat_tests.algorithms.correlation.lm]: Lagrange Multiplier tests for autocorrelation.
        - [`ts_stat_tests.algorithms.correlation.bglm`][ts_stat_tests.algorithms.correlation.bglm]: Breusch-Godfrey Lagrange Multiplier tests for residual autocorrelation.
    """
    return acorr_lm(  # type: ignore  # statsmodels' acorr_lm has incomplete type hints for these arguments
        resid=resid,
        nlags=nlags,
        store=store,
        period=period,
        ddof=ddof,
        cov_type=cov_type,
        cov_kwargs=cov_kwargs,
    )


@overload
def bglm(
    res: Union[RegressionResults, RegressionResultsWrapper],
    nlags: Optional[int] = None,
    *,
    store: Literal[False] = False,
) -> tuple[float, float, float, float]: ...
@overload
def bglm(
    res: Union[RegressionResults, RegressionResultsWrapper],
    nlags: Optional[int] = None,
    *,
    store: Literal[True],
) -> tuple[float, float, float, float, ResultsStore]: ...
@typechecked
def bglm(
    res: Union[RegressionResults, RegressionResultsWrapper],
    nlags: Optional[int] = None,
    *,
    store: bool = False,
) -> Union[
    tuple[float, float, float, float],
    tuple[float, float, float, float, ResultsStore],
]:
    """
    !!! note "Summary"

        The Breusch-Godfrey Lagrange Multiplier (BGLM) test is a statistical test used in time series forecasting to test for the presence of autocorrelation in the residuals of a model. The test is a generalization of the LM test and can be used to test for autocorrelation up to a specified order.

        This function will implement the [`acorr_breusch_godfrey()`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_breusch_godfrey.html) function from the [`statsmodels`](https://www.statsmodels.org) library.

    ???+ abstract "Details"

        BG adds lags of residual to exog in the design matrix for the auxiliary regression with residuals as endog. See Greene (2002), section 12.7.1.

        The BGLM test is performed by first fitting a time series model to the data and then obtaining the residuals from the model. The residuals are then used to estimate the autocorrelation function (ACF) up to a specified order, typically using the Box-Pierce or Ljung-Box tests. The estimated ACF values are then used to construct the BGLM test statistic, which is compared to a chi-squared distribution with degrees of freedom equal to the number of lags tested.

        The BGLM test statistic is calculated as:

        $$
        BGLM = n \\times R^2
        $$

        where:

        - $n$ is the sample size and
        - $R^2$ is the coefficient of determination from a regression of the residuals on the lagged values of the residuals and the lagged values of the predictor variable.

        ```
        BGLM = n * R^2
        ```

        Under the null hypothesis that there is no autocorrelation in the residuals of the regression model, the BGLM test statistic follows a chi-squared distribution with degrees of freedom equal to the number of lags included in the model.

        If the test statistic is greater than the critical value from the chi-squared distribution, then the null hypothesis of no autocorrelation is rejected, indicating that there is evidence of autocorrelation in the residuals. This suggests that the time series model is inadequate, and that additional terms may need to be added to the model to account for the remaining autocorrelation.

        If the test statistic is less than the critical value from the chi-squared distribution, then the null hypothesis of no autocorrelation is not rejected, indicating that there is no evidence of autocorrelation in the residuals. This suggests that the time series model is adequate, and that no further improvements are needed.

        Overall, the Breusch-Godfrey Lagrange Multiplier test is a valuable tool in time series forecasting, as it helps to assess the adequacy of a time series model and to identify areas for improvement. By testing for autocorrelation in the residuals, the test helps to ensure that the model is accurately capturing the underlying patterns in the time series data. The test is also useful for determining the appropriate order of an autoregressive integrated moving average (ARIMA) model.

        The BGLM test can be calculated using the `acorr_breusch_godfrey()` function in the `statsmodels` package in Python. The function takes a fitted regression model and the maximum number of lags to include in the test as input, and returns the BGLM test statistic and associated p-value. If the p-value of the test is less than a certain significance level (e.g. $0.05$), then there is evidence of significant autocorrelation in the residuals of the regression model up to the specified lag.

    Params:
        res (Union[RegressionResults, RegressionResultsWrapper]):
            Estimation results for which the residuals are tested for serial correlation.
        nlags (Optional[int], optional):
            Number of lags to include in the auxiliary regression. (`nlags` is highest lag).<br>
            Defaults to `None`.
        store (bool, optional):
            If `store` is `True`, then an additional class instance that contains intermediate results is returned.<br>
            Defaults to `False`.

    Returns:
        lm (float):
            Lagrange multiplier test statistic.
        lmpval (float):
            The `p-value` for Lagrange multiplier test.
        fval (float):
            The value of the `f-statistic` for F test, alternative version of the same test based on F test for the parameter restriction.
        fpval (float):
            The `p-value` of the F test.
        res_store (Optional[ResultsStore]):
            A class instance that holds intermediate results. Only returned if `store=True`.

    !!! success "Credit"
        - All credit goes to the [`statsmodels`](https://www.statsmodels.org/) library.

    !!! example "Examples"

        ```pycon {.py .python linenums="1" title="Test for Breusch-Godfrey Lagrange Multiplier in residual autocorrelation"}
        >>> from statsmodels import api as sm
        >>> from ts_stat_tests.algorithms.correlation import bglm
        >>> y = sm.datasets.longley.load_pandas().endog
        >>> X = sm.datasets.longley.load_pandas().exog
        >>> X = sm.add_constant(X)
        >>> res_lm, res_lmpval, res_fval, res_fpval = bglm(sm.OLS(y, X).fit())
        >>> print(res_lm)
        5.1409448555268185
        >>> print(res_lmpval)
        0.16176265367835008
        >>> print(res_fval)
        0.9468493873718188
        >>> print(res_fpval)
        0.4751521243357578
        ```

    ??? question "References"
        1. Greene, W. H. Econometric Analysis. New Jersey. Prentice Hall; 5th edition. (2002).

    ??? tip "See Also"
        - [`statsmodels.regression.linear_model.OLS.fit`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.fit.html): Fit a linear model.
        - [`statsmodels.regression.linear_model.RegressionResults`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html): The output results of a linear regression model.
        - [`statsmodels.stats.diagnostic.het_arch`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_arch.html#statsmodels.stats.diagnostic.het_arch): Conditional heteroskedasticity testing.
        - [`statsmodels.stats.diagnostic.acorr_ljungbox`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html): Ljung-Box test for serial correlation.
        - [`statsmodels.stats.diagnostic.acorr_lm`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_lm.html): Lagrange Multiplier tests for autocorrelation.
        - [`statsmodels.stats.diagnostic.acorr_breusch_godfrey`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_breusch_godfrey.html): Breusch-Godfrey test for serial correlation.
        - [`ts_stat_tests.algorithms.correlation.lb`][ts_stat_tests.algorithms.correlation.lb]: Ljung-Box test of autocorrelation in residuals.
        - [`ts_stat_tests.algorithms.correlation.lm`][ts_stat_tests.algorithms.correlation.lm]: Lagrange Multiplier tests for autocorrelation.
        - [`ts_stat_tests.algorithms.correlation.bglm`][ts_stat_tests.algorithms.correlation.bglm]: Breusch-Godfrey Lagrange Multiplier tests for residual autocorrelation.
    """
    return acorr_breusch_godfrey(  # type: ignore  # statsmodels typing for acorr_breusch_godfrey is incomplete/incompatible with our RegressionResults types
        res=res,
        nlags=nlags,
        store=store,
    )
