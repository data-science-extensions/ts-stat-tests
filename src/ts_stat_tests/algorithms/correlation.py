# ============================================================================ #
#                                                                              #
#     Title: Correlation Algorithms                                            #
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
from numpy.typing import ArrayLike, NDArray
from statsmodels.regression.linear_model import (
    RegressionResults,
    RegressionResultsWrapper,
)
from statsmodels.stats.api import acorr_breusch_godfrey, acorr_ljungbox, acorr_lm
from statsmodels.stats.diagnostic import ResultsStore
from statsmodels.tsa.api import acf as st_acf, ccf as st_ccf, pacf as st_pacf
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


VALID_LM_COV_TYPE_OPTIONS = Literal["nonrobust", "HC0", "HC1", "HC2", "HC3"]


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
) -> NDArray[np.float64]: ...
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
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
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
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
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
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
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
    NDArray[np.float64],
    tuple[NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
]:
    r"""
    !!! note "Summary"

        The autocorrelation function (ACF) is a statistical tool used to study the correlation between a time series and its lagged values. In time series forecasting, the ACF is used to identify patterns and relationships between values in a time series at different lags, which can then be used to make predictions about future values.

        This function will implement the [`acf()`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html) function from the [`statsmodels`](https://www.statsmodels.org) library.

    ???+ abstract "Details"

        The acf at lag `0` (ie., `1`) is returned.

        For very long time series it is recommended to use `fft` convolution instead. When `fft` is `False` uses a simple, direct estimator of the autocovariances that only computes the first $nlags + 1$ values. This can be much faster when the time series is long and only a small number of autocovariances are needed.

        If `adjusted` is `True`, the denominator for the autocovariance is adjusted for the loss of data.

        The ACF measures the correlation between a time series and its lagged values at different lags. The correlation is calculated as the ratio of the covariance between the series and its lagged values to the product of their standard deviations. The ACF is typically plotted as a graph, with the lag on the `x`-axis and the correlation coefficient on the `y`-axis.

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
            Number of lags to return autocorrelation for. If not provided, uses $\min(10 \times \log_{10}(n_{obs}), n_{obs}-1)$ (calculated with: `min(int(10 * np.log10(nobs)), nobs - 1)`). The returned value includes lag $0$ (ie., $1$) so size of the acf vector is $(nlags + 1,)$.<br>
            Defaults to `None`.
        qstat (bool, optional):
            If `True`, also returns the Ljung-Box $Q$ statistic and corresponding p-values for each autocorrelation coefficient; see the *Returns* section for details.<br>
            Defaults to `False`.
        fft (bool, optional):
            If `True`, computes the ACF via FFT.<br>
            Defaults to `True`.
        alpha (Optional[float], optional):
            If a number is given, the confidence intervals for the given level are returned. For instance if `alpha=0.05`, a $95\%$ confidence intervals are returned where the standard deviation is computed according to Bartlett's formula.<br>
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
        (Union[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]):
            Depending on `qstat` and `alpha`, returns the following values:
            - `acf` (NDArray[np.float64]): The autocorrelation function for lags `0, 1, ..., nlags`.
            - `confint` (NDArray[np.float64], optional): Confidence intervals for the ACF (returned if `alpha` is not `None`).
            - `qstat` (NDArray[np.float64], optional): The Ljung-Box Q-Statistic (returned if `qstat` is `True`).
            - `pvalues` (NDArray[np.float64], optional): P-values associated with the Q-statistics (returned if `qstat` is `True`).

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.correlation import acf
        >>> from ts_stat_tests.utils.data import data_airline, data_macrodata
        >>> data_macro = data_macrodata.realgdp.values
        >>> data_airline = data_airline.values

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Basic ACF"}
        >>> res_acf = acf(data_macro, nlags=5)
        >>> print(res_acf[1:6])
        [0.98685781 0.97371846 0.96014366 0.94568545 0.93054425]

        ```

        ```pycon {.py .python linenums="1" title="Example 2: ACF with Confidence Intervals and Q-Statistics"}
        >>> res_acf, res_confint, res_qstat, res_pvalues = acf(
        ...     data_macro, nlags=5, qstat=True, alpha=0.05
        ... )
        >>> print(res_acf[1:6])
        [0.98685781 0.97371846 0.96014366 0.94568545 0.93054425]
        >>> print(res_confint[1:6])
        [[0.84929531 1.12442032]
         [0.73753616 1.20990077]
         [0.65738012 1.2629072 ]
         [0.5899385  1.30143239]
         [0.53004062 1.33104787]]
        >>> print(res_qstat[:5])
        [200.63546275 396.93562234 588.75493948 775.77587865 957.77058934]
        >>> print(res_pvalues[:5])
        [1.51761209e-045 6.40508316e-087 2.76141970e-127 1.35591614e-166
         8.33354393e-205]

        ```

        ```pycon {.py .python linenums="1" title="Example 3: ACF without FFT"}
        >>> res_acf, res_confint, res_qstat, res_pvalues = acf(
        ...     data_macro, nlags=5, qstat=True, alpha=0.05, fft=False
        ... )
        >>> print(res_acf[1:6])
        [0.98685781 0.97371846 0.96014366 0.94568545 0.93054425]
        >>> print(res_qstat[:5])
        [200.63546275 396.93562234 588.75493948 775.77587865 957.77058934]

        ```

        ```pycon {.py .python linenums="1" title="Example 4: ACF with Adjusted Denominator"}
        >>> res_acf, res_confint = acf(data_macro, nlags=5, adjusted=True, alpha=0.05)
        >>> print(res_acf[1:6])
        [0.99174325 0.98340721 0.97454582 0.9646942  0.95404284]
        >>> print(res_confint[1:6])
        [[0.85418074 1.12930575]
         [0.74645168 1.22036273]
         [0.66999819 1.27909344]
         [0.60595482 1.32343358]
         [0.54917796 1.35890772]]

        ```

    ??? equation "Calculation"

        The ACF at lag $k$ is defined as:

        $$
        ACF(k) = \frac{ Cov(Y_t, Y_{t-k}) }{ \sqrt{Var(Y_t) \times Var(Y_{t-k})} }
        $$

        where:

        - $Y_t$ and $Y_{t-k}$ are the values of the time series at time $t$ and time $t-k$, respectively,
        - $Cov(Y_t, Y_{t-k})$ is the covariance between the two values, and
        - $Var(Y_t)$ and $Var(Y_{t-k})$ are the variances of the two values.

        For a stationary series, this simplifies to:

        $$
        ACF(k) = \frac{ Cov(Y_t, Y_{t-k}) }{ Var(Y_t) }
        $$

    ??? success "Credit"
        - All credit goes to the [`statsmodels`](https://www.statsmodels.org/) library.

    ??? question "References"
        1. Parzen, E., 1963. On spectral analysis with missing observations and amplitude modulation. Sankhya: The Indian Journal of Statistics, Series A, pp.383-392.
        2. Brockwell and Davis, 1987. Time Series Theory and Methods.
        3. Brockwell and Davis, 2010. Introduction to Time Series and Forecasting, 2nd edition.

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
) -> NDArray[np.float64]: ...
@overload
def pacf(
    x: ArrayLike1D,
    nlags: Optional[int] = None,
    method: VALID_PACF_METHOD_OPTIONS = "ywadjusted",
    *,
    alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
@typechecked
def pacf(
    x: ArrayLike1D,
    nlags: Optional[int] = None,
    method: VALID_PACF_METHOD_OPTIONS = "ywadjusted",
    *,
    alpha: Optional[float] = None,
) -> Union[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]]:
    r"""
    !!! note "Summary"

        The partial autocorrelation function (PACF) is a statistical tool used in time series forecasting to identify the direct relationship between two variables, controlling for the effect of the other variables in the time series. In other words, the PACF measures the correlation between a time series and its lagged values, while controlling for the effects of other intermediate lags.

        This function will implement the [`pacf()`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.pacf.html) function from the [`statsmodels`](https://www.statsmodels.org) library.

    ???+ abstract "Details"

        Based on simulation evidence across a range of low-order ARMA models, the best methods based on root MSE are Yule-Walker (MLW), Levinson-Durbin (MLE) and Burg, respectively. The estimators with the lowest bias included these three in addition to OLS and OLS-adjusted. Yule-Walker (adjusted) and Levinson-Durbin (adjusted) performed consistently worse than the other options.

        The PACF is a plot of the correlation between a time series and its lagged values, controlling for the effect of other lags. The PACF is useful for identifying the order of an autoregressive (AR) model, which is a type of model used in time series forecasting. The order of an AR model is the number of lags that are used to predict future values.

        The PACF is calculated using the Yule-Walker equations, which are a set of linear equations that describe the relationship between a time series and its lagged values. The PACF is calculated as the difference between the correlation coefficient at lag $k$ and the correlation coefficient at lag $k-1$, controlling for the effects of intermediate lags.

        The PACF is typically plotted as a graph, with the lag on the `x`-axis and the correlation coefficient on the `y`-axis. If the PACF shows a strong positive correlation at lag $k$, this means that values in the time series at time $t$ and time $t-k$ are strongly related, after controlling for the effects of intermediate lags. This suggests that past values can be used to predict future values using an AR model with an order of $k$.

        Overall, the partial autocorrelation function is a valuable tool in time series forecasting, as it helps to identify the order of an autoregressive model and to control for the effects of intermediate lags. By identifying the direct relationship between two variables, the PACF can help to improve the accuracy of time series forecasting models.

        The PACF can be calculated using the `pacf()` function in the `statsmodels` package in Python. The function takes a time series array as input and returns an array of partial autocorrelation coefficients at different lags. The significance of the partial autocorrelation coefficients can be tested using the same Ljung-Box test as for the ACF. If the p-value of the test is less than a certain significance level (e.g. $0.05$), then there is evidence of significant partial autocorrelation in the time series up to the specified lag.

    Params:
        x (ArrayLike1D):
            Observations of time series for which pacf is calculated.
        nlags (Optional[int], optional):
            Number of lags to return autocorrelation for. If not provided, uses $\min(10 \times \log_{10}(n_{obs}), \lfloor \frac{n_{obs}}{2} \rfloor - 1)$ (calculated with: `min(int(10*np.log10(nobs)), nobs // 2 - 1)`). The returned value includes lag `0` (ie., `1`) so size of the pacf vector is $(nlags + 1,)$.<br>
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
        (Union[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]]):
            Depending on `alpha`, returns the following values:
            - `pacf` (NDArray[np.float64]): The partial autocorrelations for lags `0, 1, ..., nlags`.
            - `confint` (NDArray[np.float64], optional): Confidence intervals for the PACF (returned if `alpha` is not `None`).

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.correlation import pacf
        >>> from ts_stat_tests.utils.data import data_airline
        >>> data = data_airline.values

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Basic PACF using Yule-Walker adjusted"}
        >>> res_pacf = pacf(data, nlags=5)
        >>> print(res_pacf[1:6])
        [ 0.95467704 -0.26527732  0.05546955  0.10885622  0.08112579]

        ```

        ```pycon {.py .python linenums="1" title="Example 2: PACF with confidence intervals"}
        >>> res_pacf, res_confint = pacf(data, nlags=5, alpha=0.05)
        >>> print(res_confint[1:3])
        [[ 0.79134671  1.11800737]
         [-0.42860765 -0.10194698]]

        ```

        ```pycon {.py .python linenums="1" title="Example 3: PACF using OLS method"}
        >>> res_pacf_ols = pacf(data, nlags=5, method="ols")
        >>> print(res_pacf_ols[1:6])
        [ 0.95893198 -0.32983096  0.2018249   0.14500798  0.25848232]

        ```

        ```pycon {.py .python linenums="1" title="Example 4: PACF using Levinson-Durbin recursion with bias correction"}
        >>> res_pacf_ld = pacf(data, nlags=5, method="ldadjusted")
        >>> print(res_pacf_ld[1:6])
        [ 0.95467704 -0.26527732  0.05546955  0.10885622  0.08112579]

        ```

    ??? equation "Calculation"

        The PACF at lag $k$ is defined as:

        $$
        PACF(k) = \text{Corr}\left( Y_t, Y_{t-k} \mid Y_{t-1}, Y_{t-2}, \dots, Y_{t-k+1} \right)
        $$

        where:

        - $Y_t$ and $Y_{t-k}$ are the values of the time series at time $t$ and time $t-k$, respectively, and
        - $Y_{t-1}, Y_{t-2}, \dots, Y_{t-k+1}$ are the values of the time series at intervening lags.
        - $\text{Corr}()$ denotes the correlation coefficient between two variables.

    ??? success "Credit"
        - All credit goes to the [`statsmodels`](https://www.statsmodels.org/) library.

    ??? question "References"
        1. Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: forecasting and control. John Wiley & Sons, p. 66.
        2. Brockwell, P.J. and Davis, R.A., 2016. Introduction to time series and forecasting. Springer.

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
) -> NDArray[np.float64]: ...
@overload
def ccf(
    x: ArrayLike,
    y: ArrayLike,
    adjusted: bool = True,
    fft: bool = True,
    *,
    nlags: Optional[int] = None,
    alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
@typechecked
def ccf(
    x: ArrayLike,
    y: ArrayLike,
    adjusted: bool = True,
    fft: bool = True,
    *,
    nlags: Optional[int] = None,
    alpha: Optional[float] = None,
) -> Union[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]]:
    r"""
    !!! note "Summary"

        The cross-correlation function (CCF) is a statistical tool used in time series forecasting to measure the correlation between two time series at different lags. It is used to study the relationship between two time series, and can help to identify lead-lag relationships and causal effects.

        This function will implement the [`ccf()`](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.ccf.html) function from the [`statsmodels`](https://www.statsmodels.org) library.

    ???+ abstract "Details"

        If `adjusted` is `True`, the denominator for the autocovariance is adjusted.

        The CCF measures the correlation between two time series at different lags. It is calculated as the ratio of the covariance between the two series at lag $k$ to the product of their standard deviations. The CCF is typically plotted as a graph, with the lag on the `x`-axis and the correlation coefficient on the `y`-axis.

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
        (Union[NDArray[np.float64], tuple[NDArray[np.float64], NDArray[np.float64]]]):
            Depending on `alpha`, returns the following values:
            - `ccf` (NDArray[np.float64]): The cross-correlation function of `x` and `y` for lags `0, 1, ..., nlags`.
            - `confint` (NDArray[np.float64], optional): Confidence intervals for the CCF (returned if `alpha` is not `None`).

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.correlation import ccf
        >>> from ts_stat_tests.utils.data import data_airline
        >>> data = data_airline.values

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Basic CCF"}
        >>> res = ccf(data, data + 1, adjusted=True)
        >>> print(res[:5])
        [1.         0.95467704 0.88790688 0.82384458 0.774129  ]

        ```

        ```pycon {.py .python linenums="1" title="Example 2: CCF with confidence intervals"}
        >>> res_ccf, res_conf = ccf(data, data + 1, alpha=0.05)
        >>> print(res_ccf[:5])
        [1.         0.95467704 0.88790688 0.82384458 0.774129  ]
        >>> print(res_conf[:5])
        [[0.83666967 1.16333033]
         [0.79134671 1.11800737]
         [0.72457654 1.05123721]
         [0.66051425 0.98717492]
         [0.61079867 0.93745933]]

        ```

        ```pycon {.py .python linenums="1" title="Example 3: CCF without adjustment"}
        >>> res_ccf_no_adj = ccf(data, data + 1, adjusted=False)
        >>> print(res_ccf_no_adj[:5])
        [1.         0.94804734 0.87557484 0.80668116 0.75262542]

        ```

        ```pycon {.py .python linenums="1" title="Example 4: CCF without FFT"}
        >>> res_ccf_no_fft = ccf(data, data + 1, fft=False)
        >>> print(res_ccf_no_fft[:5])
        [1.         0.95467704 0.88790688 0.82384458 0.774129  ]

        ```

    ??? equation "Calculation"

        The CCF at lag $k$ is defined as:

        $$
        CCF(k) = \text{Corr}(X_t, Y_{t-k})
        $$

        where:

        - $X_t$ and $Y_{t-k}$ are the values of the two time series at time $t$ and time $t-k$, respectively.
        - $\text{Corr}()$ denotes the correlation coefficient between two variables.

    ??? success "Credit"
        - All credit goes to the [`statsmodels`](https://www.statsmodels.org/) library.

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

        Under the null hypothesis, the test statistic follows a chi-squared distribution with degrees of freedom equal to $m-p$, where $p$ is the number of parameters estimated in fitting the time series model.

        The Ljung-Box test is performed by calculating the autocorrelation function (ACF) of the residuals from a time series model, and then comparing the ACF values to the expected values under the null hypothesis of no autocorrelation. The test statistic is calculated as the sum of the squared autocorrelations up to a given lag, and is compared to a chi-squared distribution with degrees of freedom equal to the number of lags tested.

        If the test statistic is greater than the critical value from the chi-squared distribution, then the null hypothesis of no autocorrelation is rejected, indicating that there is evidence of autocorrelation in the residuals. This suggests that the time series model is inadequate, and that additional terms may need to be added to the model to account for the remaining autocorrelation.

        If the test statistic is less than the critical value from the chi-squared distribution, then the null hypothesis of no autocorrelation is not rejected, indicating that there is no evidence of autocorrelation in the residuals. This suggests that the time series model is adequate, and that no further improvements are needed.

        Overall, the Ljung-Box test is a valuable tool in time series forecasting, as it helps to assess the adequacy of a time series model and to identify areas for improvement. By testing for autocorrelation in the residuals, the test helps to ensure that the model is accurately capturing the underlying patterns in the time series data.

        The Ljung-Box test can be calculated using the `acorr_ljungbox()` function in the `statsmodels` package in Python. The function takes a time series array and the maximum lag $m$ as input, and returns an array of $Q$-statistics and associated p-values for each lag up to $m$. If the p-value of the test is less than a certain significance level (e.g. $0.05$), then there is evidence of significant autocorrelation in the time series up to the specified lag.

    Params:
        x (ArrayLike):
            The data series. The data is demeaned before the test statistic is computed.
        lags (Optional[Union[int, ArrayLike]], optional):
            If lags is an integer (`int`) then this is taken to be the largest lag that is included, the test result is reported for all smaller lag length. If lags is a list or array, then all lags are included up to the largest lag in the list, however only the tests for the lags in the list are reported. If lags is `None`, then the default maxlag is currently $\min(\lfloor \frac{n_{obs}}{2} \rfloor - 2, 40)$ (calculated with: `min(nobs // 2 - 2, 40)`). The default number of `lags` changes if `period` is set.
            !!! deprecation "Deprecation"
                After `statsmodels` version `0.12`, this calculation will change from

                $$
                \min\left(\lfloor \frac{n_{obs}}{2} \rfloor - 2, 40\right)
                $$

                to

                $$
                \min\left(10, \frac{n_{obs}}{5}\right)
                $$

            Defaults to `None`.
        boxpierce (bool, optional):
            If `True`, then additional to the results of the Ljung-Box test also the Box-Pierce test results are returned.<br>
            Defaults to `False`.
        model_df (int, optional):
            Number of degrees of freedom consumed by the model. In an ARMA model, this value is usually $p+q$ where $p$ is the AR order and $q$ is the MA order. This value is subtracted from the degrees-of-freedom used in the test so that the adjusted dof for the statistics are $lags - model_df$. If $lags - model_df \le 0$, then `NaN` is returned.<br>
            Defaults to `0`.
        period (Optional[int], optional):
            The period of a Seasonal time series. Used to compute the max lag for seasonal data which uses $\min(2 \times period, \lfloor \frac{n_{obs}}{5} \rfloor)$ (calculated with: `min(2*period,nobs//5)`) if set. If `None`, then the default rule is used to set the number of lags. When set, must be $\ge 2$.<br>
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
        (Union[pd.DataFrame, tuple[NDArray[np.float64], NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]):
            Depending on `return_df` and `boxpierce`, returns the following values:
            - `lb_stat` (NDArray[np.float64]): The Ljung-Box test statistic.
            - `lb_pvalue` (NDArray[np.float64]): The p-value for the Ljung-Box test.
            - `bp_stat` (NDArray[np.float64], optional): The Box-Pierce test statistic (returned if `boxpierce` is `True`).
            - `bp_pvalue` (NDArray[np.float64], optional): The p-value for the Box-Pierce test (returned if `boxpierce` is `True`).

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from statsmodels import api as sm
        >>> from ts_stat_tests.algorithms.correlation import lb
        >>> from ts_stat_tests.utils.data import data_airline
        >>> data = data_airline.values
        >>> res = sm.tsa.ARIMA(data, order=(1, 0, 1)).fit()

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Ljung-Box test on ARIMA residuals"}
        >>> results = lb(res.resid, lags=[10], return_df=True)
        >>> print(results)
              lb_stat  lb_pvalue
        10  13.844361    0.18021

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Ljung-Box and Box-Pierce tests with multiple lags"}
        >>> results = lb(res.resid, lags=[5, 10, 15], boxpierce=True, return_df=True)
        >>> print(results)
              lb_stat     lb_pvalue    bp_stat     bp_pvalue
        5    6.274986  2.803736e-01   6.019794  3.042976e-01
        10  13.844361  1.802099e-01  13.080554  2.192028e-01
        15  86.182531  5.083111e-12  78.463124  1.332482e-10

        ```

        ```pycon {.py .python linenums="1" title="Example 3: Ljung-Box test with specific lag"}
        >>> results = lb(res.resid, lags=[5], return_df=True)
        >>> print(results)
            lb_stat  lb_pvalue
        5  6.274986   0.280374

        ```

    ??? equation "Calculation"

        The Ljung-Box test statistic is calculated as:

        $$
        Q(m) = n(n+2) \sum_{k=1}^m \frac{r_k^2}{n-k}
        $$

        where:

        - $n$ is the sample size,
        - $m$ is the maximum lag being tested,
        - $r_k$ is the sample autocorrelation at lag $k$, and
        - $\sum$ denotes the sum over $k$ from $1$ to $m$.

    ??? success "Credit"
        - All credit goes to the [`statsmodels`](https://www.statsmodels.org/) library.

    ??? question "References"
        1. Green, W. "Econometric Analysis," 5th ed., Pearson, 2003.
        2. J. Carlos Escanciano, Ignacio N. Lobato "An automatic Portmanteau test for serial correlation"., Volume 151, 2009.

    ??? tip "See Also"
        - [`statsmodels.regression.linear_model.OLS.fit`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.fit.html): Fit a linear model.
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
    cov_type: VALID_LM_COV_TYPE_OPTIONS = "nonrobust",
    cov_kwargs: Optional[dict] = None,
) -> tuple[float, NDArray[np.float64], float, float]: ...
@overload
def lm(
    resid: ArrayLike,
    nlags: Optional[int] = None,
    *,
    store: Literal[True],
    period: Optional[int] = None,
    ddof: int = 0,
    cov_type: VALID_LM_COV_TYPE_OPTIONS = "nonrobust",
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
    cov_type: VALID_LM_COV_TYPE_OPTIONS = "nonrobust",
    cov_kwargs: Optional[dict] = None,
) -> Union[
    tuple[float, Union[float, NDArray[np.float64]], float, float],
    tuple[float, Union[float, NDArray[np.float64]], float, float, ResultsStore],
]:
    r"""
    !!! note "Summary"

        The Lagrange Multiplier (LM) test is a statistical test used in time series forecasting to test for the presence of autocorrelation in a model. The test is based on the residual sum of squares (RSS) of a time series model, and can be used to assess the adequacy of the model and to identify areas for improvement.

        This function implements the [`acorr_lm()`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_lm.html) function from the [`statsmodels`](https://www.statsmodels.org) library.

    ???+ abstract "Details"

        This is a generic Lagrange Multiplier (LM) test for autocorrelation. It returns Engle's ARCH test if `resid` is the squared residual array. The Breusch-Godfrey test is a variation on this LM test with additional exogenous variables in the auxiliary regression.

        The LM test proceeds by:

        - Fitting a time series model to the data and obtaining the residuals.
        - Running an auxiliary regression of these residuals on their past `nlags` values (and any relevant exogenous variables).
        - Computing the LM statistic as $(n_{obs} - ddof) \times R^2$ from this auxiliary regression.

        Under the null hypothesis that the autocorrelations up to the specified lag are zero (no serial correlation in the residuals), the LM statistic is asymptotically distributed as a chi-squared random variable with degrees of freedom equal to the number of lagged residual terms included in the auxiliary regression (i.e. the number of lags being tested, adjusted for any restrictions implied by the model).

        If the test statistic is greater than the critical value from the chi-squared distribution (or equivalently, if the p-value is less than a chosen significance level such as $0.05$), then the null hypothesis of no autocorrelation is rejected, indicating that there is evidence of autocorrelation in the residuals.

        The LM test is a generalization of the Durbin-Watson test, which is a simpler test that only tests for first-order autocorrelation.

    Params:
        resid (ArrayLike):
            Time series to test.
        nlags (Optional[int], optional):
            Highest lag to use. Defaults to `None`.
            !!! deprecation "Deprecation"
                The behavior of this parameter will change after `statsmodels` version `0.12`.
        store (bool, optional):
            If `True` then the intermediate results are also returned. Defaults to `False`.
        period (Optional[int], optional):
            The period of a Seasonal time series. Used to compute the max lag for seasonal data which uses $\min(2 \times period, \lfloor \frac{n_{obs}}{5} \rfloor)$ (calculated with: `min(2*period,nobs//5)`) if set. If `None`, then the default rule is used to set the number of lags. When set, must be $\ge 2$. Defaults to `None`.
        ddof (int, optional):
            The number of degrees of freedom consumed by the model used to produce `resid`. Defaults to `0`.
        cov_type (VALID_LM_COV_TYPE_OPTIONS, optional):
            Covariance type. The default is `"nonrobust"` which uses the classic OLS covariance estimator. Specify one of `"HC0"`, `"HC1"`, `"HC2"`, `"HC3"` to use White's covariance estimator. All covariance types supported by `OLS.fit` are accepted. Defaults to `"nonrobust"`.
        cov_kwargs (Optional[dict], optional):
            Dictionary of covariance options passed to `OLS.fit`. See [`OLS.fit`](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.fit.html) for more details. Defaults to `None`.

    Returns:
        (Union[tuple[float, Union[float, NDArray[np.float64]], float, float], tuple[float, Union[float, NDArray[np.float64]], float, float, ResultsStore]]):
            Returns the following values:
            - `lm` (float): Lagrange multiplier test statistic.
            - `lmpval` (Union[float, NDArray[np.float64]]): The p-value for the Lagrange multiplier test.
            - `fval` (float): The f-statistic of the F test.
            - `fpval` (float): The p-value of the F test.
            - `res_store` (ResultsStore, optional): Intermediate results (returned if `store` is `True`).

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.correlation import lm
        >>> from ts_stat_tests.utils.data import data_airline
        >>> data = data_airline.values

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Lagrange Multiplier test"}
        >>> res_lm, res_p, res_f, res_fp = lm(data)
        >>> print(f"LM statistic: {res_lm:.4f}")
        LM statistic: 128.0966
        >>> print(f"p-value: {res_p:.4e}")
        p-value: 1.1417e-22

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Lagrange Multiplier test with intermediate results"}
        >>> res_lm, res_p, res_f, res_fp, res_store = lm(data, store=True)
        >>> print(f"LM statistic: {res_lm:.4f}")
        LM statistic: 128.0966
        >>> print(f"p-value: {res_p:.4e}")
        p-value: 1.1417e-22

        ```

        ```pycon {.py .python linenums="1" title="Example 3: Lagrange Multiplier test with robust covariance"}
        >>> res_lm, res_p, res_f, res_fp = lm(data, cov_type="HC3")
        >>> print(f"LM statistic: {res_lm:.4f}")
        LM statistic: 2063.3981
        >>> print(f"p-value: {res_p:.1f}")
        p-value: 0.0

        ```

        ```pycon {.py .python linenums="1" title="Example 4: Lagrange Multiplier test with seasonal period"}
        >>> res_lm, res_p, res_f, res_fp = lm(data, period=12)
        >>> print(f"LM statistic: {res_lm:.4f}")
        LM statistic: 119.1109
        >>> print(f"p-value: {res_p:.4e}")
        p-value: 1.3968e-14

        ```

        ```pycon {.py .python linenums="1" title="Example 5: Lagrange Multiplier test with specified degrees of freedom"}
        >>> res_lm, res_p, res_f, res_fp = lm(data, ddof=2)
        >>> print(f"LM statistic: {res_lm:.4f}")
        LM statistic: 126.1847
        >>> print(f"p-value: {res_p:.4e}")
        p-value: 2.7990e-22

        ```

    ??? equation "Calculation"

        The LM test statistic is computed as:

        $$
        LM = (n_{obs} - ddof) \times R^2
        $$

        where:

        - $R^2$ is the coefficient of determination from the auxiliary regression of the residuals on their own `nlags` lags,
        - $n_{obs}$ is the number of observations, and
        - $ddof$ is the model degrees of freedom lost due to parameter estimation.

    ??? success "Credit"
        - All credit goes to the [`statsmodels`](https://www.statsmodels.org/) library.

    ??? tip "See Also"
        - [`statsmodels.stats.diagnostic.acorr_lm`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_lm.html): Lagrange Multiplier tests for autocorrelation.
        - [`ts_stat_tests.algorithms.correlation.lb`][ts_stat_tests.algorithms.correlation.lb]: Ljung-Box test of autocorrelation in residuals.
        - [`ts_stat_tests.algorithms.correlation.bglm`][ts_stat_tests.algorithms.correlation.bglm]: Breusch-Godfrey Lagrange Multiplier tests for residual autocorrelation.
    """
    return acorr_lm(
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
) -> tuple[float, Union[float, NDArray[np.float64]], float, float]: ...
@overload
def bglm(
    res: Union[RegressionResults, RegressionResultsWrapper],
    nlags: Optional[int] = None,
    *,
    store: Literal[True],
) -> tuple[float, Union[float, NDArray[np.float64]], float, float, ResultsStore]: ...
@typechecked
def bglm(
    res: Union[RegressionResults, RegressionResultsWrapper],
    nlags: Optional[int] = None,
    *,
    store: bool = False,
) -> Union[
    tuple[float, Union[float, NDArray[np.float64]], float, float],
    tuple[float, Union[float, NDArray[np.float64]], float, float, ResultsStore],
]:
    r"""
    !!! note "Summary"

        The Breusch-Godfrey Lagrange Multiplier (BGLM) test is a statistical test used in time series forecasting to test for the presence of autocorrelation in the residuals of a model. The test is a generalization of the LM test and can be used to test for autocorrelation up to a specified order.

        This function implements the [`acorr_breusch_godfrey()`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_breusch_godfrey.html) function from the [`statsmodels`](https://www.statsmodels.org) library.

    ???+ abstract "Details"

        BG adds lags of residual to exog in the design matrix for the auxiliary regression with residuals as endog. See Greene (2002), section 12.7.1.

        The BGLM test is performed by first fitting a time series model to the data and then obtaining the residuals from the model. The residuals are then used to estimate the autocorrelation function (ACF) up to a specified order.

        Under the null hypothesis that there is no autocorrelation in the residuals of the regression model, the BGLM test statistic follows a chi-squared distribution with degrees of freedom equal to the number of lags included in the model.

        If the test statistic is greater than the critical value from the chi-squared distribution, then the null hypothesis of no autocorrelation is rejected, indicating that there is evidence of autocorrelation in the residuals.

    Params:
        res (Union[RegressionResults, RegressionResultsWrapper]):
            Estimation results for which the residuals are tested for serial correlation.
        nlags (Optional[int], optional):
            Number of lags to include in the auxiliary regression. (`nlags` is highest lag). Defaults to `None`.
        store (bool, optional):
            If `store` is `True`, then an additional class instance that contains intermediate results is returned. Defaults to `False`.

    Returns:
        (Union[tuple[float, float, float, float], tuple[float, float, float, float, ResultsStore]]):
            Returns the following values:
            - `lm` (float): Lagrange multiplier test statistic.
            - `lmpval` (float): The p-value for the Lagrange multiplier test.
            - `fval` (float): The value of the f-statistic for the F test.
            - `fpval` (float): The p-value of the F test.
            - `res_store` (ResultsStore, optional): Intermediate results (returned if `store` is `True`).

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from statsmodels import api as sm
        >>> from ts_stat_tests.algorithms.correlation import bglm
        >>> y = sm.datasets.longley.load_pandas().endog
        >>> X = sm.datasets.longley.load_pandas().exog
        >>> X = sm.add_constant(X)
        >>> model = sm.OLS(y, X).fit()

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Breusch-Godfrey test"}
        >>> res_lm, res_p, res_f, res_fp = bglm(model)
        >>> print(f"LM statistic: {res_lm:.4f}")
        LM statistic: 5.1409
        >>> print(f"p-value: {res_p:.4f}")
        p-value: 0.1618

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Breusch-Godfrey test with intermediate results"}
        >>> res_lm, res_p, res_f, res_fp, res_store = bglm(model, store=True)
        >>> print(f"LM statistic: {res_lm:.4f}")
        LM statistic: 5.1409
        >>> print(f"p-value: {res_p:.4f}")
        p-value: 0.1618

        ```

        ```pycon {.py .python linenums="1" title="Example 3: Breusch-Godfrey test with specified lags"}
        >>> res_lm, res_p, res_f, res_fp = bglm(model, nlags=2)
        >>> print(f"LM statistic: {res_lm:.4f}")
        LM statistic: 2.8762
        >>> print(f"p-value: {res_p:.4f}")
        p-value: 0.2374

        ```

    ??? equation "Calculation"

        The BGLM test statistic is calculated as:

        $$
        BGLM = n \times R^2
        $$

        where:

        - $n$ is the sample size and
        - $R^2$ is the coefficient of determination from a regression of the residuals on the lagged values of the residuals and the lagged values of the predictor variable.

    ??? success "Credit"
        - All credit goes to the [`statsmodels`](https://www.statsmodels.org/) library.

    ??? question "References"
        1. Greene, W. H. Econometric Analysis. New Jersey. Prentice Hall; 5th edition. (2002).

    ??? tip "See Also"
        - [`statsmodels.stats.diagnostic.acorr_breusch_godfrey`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_breusch_godfrey.html): Breusch-Godfrey test for serial correlation.
        - [`ts_stat_tests.algorithms.correlation.lb`][ts_stat_tests.algorithms.correlation.lb]: Ljung-Box test of autocorrelation in residuals.
        - [`ts_stat_tests.algorithms.correlation.lm`][ts_stat_tests.algorithms.correlation.lm]: Lagrange Multiplier tests for autocorrelation.
    """
    return acorr_breusch_godfrey(
        res=res,
        nlags=nlags,
        store=store,
    )
