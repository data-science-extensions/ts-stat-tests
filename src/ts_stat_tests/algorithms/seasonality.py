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
from numpy.typing import ArrayLike, NDArray
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
    r"""
    !!! note "Summary"
        The $QS$ test, also known as the Ljung-Box test, is a statistical test used to determine whether there is any seasonality present in a time series forecasting model. It is based on the autocorrelation function (ACF) of the residuals, which is a measure of how correlated the residuals are at different lags.

    ???+ abstract "Details"

        If `residuals=False` the `autoarima` settings are ignored.

        If `residuals=True`, a non-seasonal ARIMA model is estimated for the time series. And the residuals of the fitted model are used as input to the test statistic. If an automatic order selection is used, the Hyndman-Khandakar algorithm is employed with: $\max(p)=\max(q)<=3$.

        The null hypothesis is that there is no correlation in the residuals beyond the specified lags, indicating no seasonality. The alternative hypothesis is that there is significant correlation, indicating seasonality.

        Here are the steps for performing the $QS$ test:

        1. Fit a time series model to your data, such as an ARIMA or SARIMA model.
        1. Calculate the residuals, which are the differences between the observed values and the predicted values from the model.
        1. Calculate the ACF of the residuals.
        1. Calculate the Q statistic, which is the sum of the squared values of the autocorrelations at different lags, up to a specified lag. Using the formula above.
        1. Compare the Q statistic to the critical value from the chi-squared distribution with degrees of freedom equal to the number of lags. If the Q statistic is greater than the critical value, then the null hypothesis is rejected, indicating that there is evidence of seasonality in the residuals.

        In summary, the $QS$ test is a useful tool for determining whether a time series forecasting model has adequately accounted for seasonality in the data. By detecting any seasonality present in the residuals, it helps to ensure that the model is capturing all the important patterns in the data and making accurate predictions.

        This function will implement the Python version of the R function [`qs()`](https://rdrr.io/cran/seastests/man/qs.html) from the [`seastests`](https://cran.r-project.org/web/packages/seastests/index.html) library.

    Params:
        x (ArrayLike):
            The univariate time series data to test.
        freq (int, optional):
            The frequency of the time series data.<br>
            Default: `0`
        diff (bool, optional):
            Whether or not to run `np.diff()` over the data.<br>
            Default: `True`
        residuals (bool, optional):
            Whether or not to run & return the residuals from the function.<br>
            Default: `False`
        autoarima (bool, optional):
            Whether or not to run the `AutoARIMA()` algorithm over the data.<br>
            Default: `True`

    Raises:
        (AttributeError):
            If `x` is empty, or `freq` is too low for the data to be adequately tested.
        (ValueError):
            If, after differencing the data (by using `np.diff()`), any of the values are `None` (or `Null` or `np.nan`), then it cannot be used for QS Testing.

    Returns:
        (Union[tuple[float, float], tuple[float, float, Optional[ARIMA]]]):
            The results of the QS test.
            - stat (float): The $\text{QS}$ score for the given data set.
            - pval (float): The p-value of the given test. Calculated using the survival function of the chi-squared algorithm (also defined as $1-\text{cdf(...)}$). For more info, see: [scipy.stats.chi2](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html)
            - model (Optional[ARIMA]): The ARIMA model used in the calculation of this test. Returned if `residuals` is `True`.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.algorithms.seasonality import qs
        >>> data = load_airline().values
        >>> qs(data, freq=12)
        (194.469289..., 5.909223...)

        ```

        ```pycon {.py .python linenums="1" title="Advanced usage"}
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.algorithms.seasonality import qs
        >>> data = load_airline().values
        >>> qs(data, freq=12, diff=True, residuals=True, autoarima=True)
        The differences of the residuals of a non-seasonal ARIMA model are computed and used. It may be better to either only take the differences or use the residuals.
        (101.8592..., 7.6126..., ARIMA(order=(1, 1, 1), scoring_args={}, suppress_warnings=True))

        ```

    ??? equation "Calculation"

        The $Q$ statistic is given by:

        $$
        QS = (n \times (n+2)) \times \sum_{k=1}^{h} \frac{r_k^2}{n-k}
        $$

        where:

        - $n$ is the sample size,
        - $r_k$ is the autocorrelation at lag $k$, and
        - $h$ is the maximum lag to be considered.

        ```
        QS = n(n+2) * sum(r_k^2 / (n-k)) for k = 1 to h
        ```

    ??? success "Credit"
        - All credit goes to the [`seastests`](https://cran.r-project.org/web/packages/seastests/index.html) library.

    ??? question "References"
        1. Hyndman, R. J. and Y. Khandakar (2008). Automatic Time Series Forecasting: The forecast Package for R. Journal of Statistical Software 27 (3), 1-22.
        1. Maravall, A. (2011). Seasonality Tests and Automatic Model Identification in TRAMO-SEATS. Bank of Spain.
        1. Ollech, D. and Webel, K. (2020). A random forest-based approach to identifying the most informative seasonality tests. Deutsche Bundesbank's Discussion Paper series 55/2020.

    ??? tip "See Also"
        - [github/seastests/qs.R](https://github.com/cran/seastests/blob/master/R/qs.R)
        - [rdrr/seastests/qs](https://rdrr.io/cran/seastests/man/qs.html)
        - [rdocumentation/seastests/qs](https://www.rdocumentation.org/packages/seastests/versions/0.15.4/topics/qs)
        - [Machine Learning Mastery/How to Identify and Remove Seasonality from Time Series Data with Python](https://machinelearningmastery.com/time-series-seasonality-with-python)
        - [StackOverflow/Simple tests for seasonality in Python](https://stackoverflow.com/questions/62754218/simple-tests-for-seasonality-in-python)
    """

    _x: NDArray[np.float64] = np.asarray(x, dtype=float)
    if np.isnan(_x).all():
        raise AttributeError("All observations are NaN.")
    if diff and residuals:
        print(
            "The differences of the residuals of a non-seasonal ARIMA model are computed and used. "
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
            except (ValueError, RuntimeError, IndexError):
                try:
                    model = ARIMA(order=(0, 1, 1)).fit(y=_x)
                except (ValueError, RuntimeError, IndexError):
                    print("Could not estimate any ARIMA model, original data series is used.")
            if model is not None:
                _x = model.resid()
        else:
            try:
                model = ARIMA(order=(0, 1, 1)).fit(y=_x)
            except (ValueError, RuntimeError, IndexError):
                print("Could not estimate any ARIMA model, original data series is used.")
            if model is not None:
                _x = model.resid()

    # Do diff
    y: NDArray[np.float64] = np.diff(_x) if diff else _x

    # Pre-check
    if np.nanvar(y[~np.isnan(y)]) == 0:
        raise ValueError(
            "The Series is a constant (possibly after transformations). QS-Test cannot be computed on constants."
        )

    # Test Statistic
    acf_output: NDArray[np.float64] = _acf(x=y, nlags=freq * 2, missing="drop")
    rho_output: NDArray[np.float64] = acf_output[[freq, freq * 2]]
    rho: NDArray[np.float64] = np.array([0, 0]) if np.any(np.array(rho_output) <= 0) else rho_output
    N: int = len(y[~np.isnan(y)])
    QS: float = float(N * (N + 2) * (rho[0] ** 2 / (N - freq) + rho[1] ** 2 / (N - freq * 2)))
    Pval: float = float(chi2.sf(QS, 2))

    if residuals:
        return QS, Pval, model
    return QS, Pval


@typechecked
def ocsb(x: ArrayLike, m: int, lag_method: str = "aic", max_lag: int = 3) -> int:
    r"""
    !!! note "Summary"
        Compute the Osborn, Chui, Smith, and Birchenhall ($OCSB$) test for an input time series to determine whether it needs seasonal differencing. The regression equation may include lags of the dependent variable. When `lag_method="fixed"`, the lag order is fixed to `max_lag`; otherwise, `max_lag` is the maximum number of lags considered in a lag selection procedure that minimizes the `lag_method` criterion, which can be `"aic"`, `"bic"` or corrected AIC `"aicc"`.

    ???+ abstract "Details"

        The $OCSB$ test is a statistical test that is used to check the presence of seasonality in time series data. Seasonality refers to a pattern in the data that repeats itself at regular intervals.

        The $OCSB$ test is based on the null hypothesis that there is no seasonality in the time series data. If the p-value of the test is less than the significance level (usually $0.05$), then the null hypothesis is rejected, and it is concluded that there is seasonality in the data.

        The $OCSB$ test involves dividing the data into two halves and calculating the mean of each half. Then, the differences between the means of each pair of halves are calculated for each possible pair of halves. Finally, the mean of these differences is calculated, and a test statistic is computed.

        The $OCSB$ test is useful for testing seasonality in time series data because it can detect seasonal patterns that are not obvious in the original data. It is also a useful diagnostic tool for determining the appropriate seasonal differencing parameter in ARIMA models.

        Critical values for the test are based on simulations, which have been smoothed over to produce critical values for all seasonal periods

        The null hypothesis of the $OCSB$ test is that there is no seasonality in the time series, and the alternative hypothesis is that there is seasonality. The test statistic is compared to a critical value from a chi-squared distribution with degrees of freedom equal to the number of possible pairs of halves. If the test statistic is larger than the critical value, then the null hypothesis is rejected, and it is concluded that there is evidence of seasonality in the time series.

    Params:
        x (ArrayLike):
            The time series vector.
        m (int):
            The seasonal differencing term. For monthly data, e.g., this would be 12. For quarterly, 4, etc. For the OCSB test to work, `m` must exceed `1`.
        lag_method (str, optional):
            The lag method to use. One of (`"fixed"`, `"aic"`, `"bic"`, `"aicc"`). The metric for assessing model performance after fitting a linear model.<br>
            Default: `"aic"`
        max_lag (int, optional):
            The maximum lag order to be considered by `lag_method`.<br>
            Default: `3`

    Returns:
        (int):
            The seasonal differencing term. For different values of `m`, the OCSB statistic is compared to an estimated critical value, and returns 1 if the computed statistic is greater than the critical value, or 0 if not.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.algorithms.seasonality import ocsb
        >>> data = load_airline().values
        >>> ocsb(x=data, m=12)
        1

        ```

    ??? equation "Calculation"

        The equation for the $OCSB$ test statistic for a time series of length n is:

        $$
        OCSB = \frac{1}{(n-1)} \times \sum \left( \left( x[i] - x \left[ \frac{n}{2+i} \right] \right) - \left( x \left[ \frac{n}{2+i} \right] - x \left[ \frac{i+n}{2+1} \right] \right) \right) ^2
        $$

        where:

        - $n$ is the sample size, and
        - $x[i]$ is the $i$-th observation in the time series.

        ```
        OCSB = (1 / (n - 1)) * sum( ((x[i] - x[n/2+i]) - (x[n/2+i] - x[i+n/2+1]))^2 )
        ```

        In this equation, the time series is split into two halves, and the difference between the means of each half is calculated for each possible pair of halves. The sum of the squared differences is then divided by the length of the time series minus one to obtain the $OCSB$ test statistic.

    ??? success "Credit"
        - All credit goes to the [`pmdarima`](http://alkaline-ml.com/pmdarima/index.html) library with the implementation of [`pmdarima.arima.OCSBTest`](http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.OCSBTest.html).

    ??? question "References"
        - Osborn DR, Chui APL, Smith J, and Birchenhall CR (1988) "Seasonality and the order of integration for consumption", Oxford Bulletin of Economics and Statistics 50(4):361-377.
        - R's forecast::OCSB test source code: https://bit.ly/2QYQHno

    ??? tip "See Also"
        - [pmdarima.arima.OCSBTest](http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.OCSBTest.html)
    """
    return OCSBTest(m=m, lag_method=lag_method, max_lag=max_lag).estimate_seasonal_differencing_term(x)


@typechecked
def ch(x: ArrayLike, m: int) -> int:
    r"""
    !!! note "Summary"
        The Canova-Hansen test for seasonal differences. Canova and Hansen (1995) proposed a test statistic for the null hypothesis that the seasonal pattern is stable. The test statistic can be formulated in terms of seasonal dummies or seasonal cycles. The former allows us to identify seasons (e.g. months or quarters) that are not stable, while the latter tests the stability of seasonal cycles (e.g. cycles of period 2 and 4 quarters in quarterly data).

        !!! warning "Warning"
            This test is generally not used directly, but in conjunction with `pmdarima.arima.nsdiffs()`, which directly estimates the number of seasonal differences.

    ???+ abstract "Details"

        The $CH$ test (also known as the Canova-Hansen test) is a statistical test for detecting seasonality in time series data. It is based on the idea of comparing the goodness of fit of two models: a non-seasonal model and a seasonal model. The null hypothesis of the $CH$ test is that the time series is non-seasonal, while the alternative hypothesis is that the time series is seasonal.

        The test statistic is compared to a critical value from the chi-squared distribution with degrees of freedom equal to the difference in parameters between the two models. If the test statistic exceeds the critical value, the null hypothesis of non-seasonality is rejected in favor of the alternative hypothesis of seasonality.

        The $CH$ test is based on the following steps:

        1. Fit a non-seasonal autoregressive integrated moving average (ARIMA) model to the time series data, using a criterion such as Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC) to determine the optimal model order.
        1. Fit a seasonal ARIMA model to the time series data, using the same criterion to determine the optimal model order and seasonal period.
        1. Compute the sum of squared residuals (SSR) for both models.
        1. Compute the test statistic $CH$ using the formula above.
        1. Compare the test statistic to a critical value from the chi-squared distribution with degrees of freedom equal to the difference in parameters between the two models. If the test statistic exceeds the critical value, reject the null hypothesis of non-seasonality in favor of the alternative hypothesis of seasonality.

        The $CH$ test is a powerful test for seasonality in time series data, as it accounts for both the presence and the nature of seasonality. However, it assumes that the time series data is stationary, and it may not be effective for detecting seasonality in non-stationary or irregular time series data. Additionally, it may not work well for time series data with short seasonal periods or with low seasonal amplitudes. Therefore, it should be used in conjunction with other tests and techniques for detecting seasonality in time series data.

    Params:
        x (ArrayLike):
            The time series vector.
        m (int):
            The seasonal differencing term. For monthly data, e.g., this would be 12. For quarterly, 4, etc. For the Canova-Hansen test to work, `m` must exceed 1.

    Returns:
        (int):
            The seasonal differencing term.

            The $CH$ test defines a set of critical values:

            ```
            (0.4617146, 0.7479655, 1.0007818,
             1.2375350, 1.4625240, 1.6920200,
             1.9043096, 2.1169602, 2.3268562,
             2.5406922, 2.7391007)
            ```

            For different values of `m`, the $CH$ statistic is compared to the corresponding critical value, and returns 1 if the computed statistic is greater than the critical value, or 0 if not.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.algorithms.seasonality import ch
        >>> data = load_airline().values
        >>> ch(x=data, m=12)
        0

        ```

    ??? equation "Calculation"

        The test statistic for the $CH$ test is given by:

        $$
        CH = \frac { \left( \frac { SSRns - SSRs } { n - p - 1 } \right) } { \left( \frac { SSRs } { n - p - s - 1 } \right) }
        $$

        where:

        - $SSRns$ is the $SSR$ for the non-seasonal model,
        - $SSRs$ is the $SSR$ for the seasonal model,
        - $n$ is the sample size,
        - $p$ is the number of parameters in the non-seasonal model, and
        - $s$ is the number of parameters in the seasonal model.

        ```
        CH = [(SSRns - SSRs) / (n - p - 1)] / (SSRs / (n - p - s - 1))
        ```

    ??? note "Notes"
        This test is generally not used directly, but in conjunction with `pmdarima.arima.nsdiffs()`, which directly estimates the number of seasonal differences.

    ??? success "Credit"
        - All credit goes to the [`pmdarima`](http://alkaline-ml.com/pmdarima/index.html) library with the implementation of [`pmdarima.arima.CHTest`](http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.CHTest.html).

    ??? question "References"
        - Testing for seasonal stability using the Canova and Hansen test statistic: http://bit.ly/2wKkrZo
        - R source code for CH test: https://github.com/robjhyndman/forecast/blob/master/R/arima.R#L148

    ??? tip "See Also"
        - [`pmdarima.arima.CHTest`](http://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.CHTest.html)
    """
    return CHTest(m=m).estimate_seasonal_differencing_term(x)


@typechecked
def seasonal_strength(x: ArrayLike, m: int) -> float:
    r"""
    !!! note "Summary"
        The seasonal strength test is a statistical test for detecting the strength of seasonality in time series data. It measures the extent to which the seasonal component of a time series explains the variation in the data.

    ???+ abstract "Details"

        The seasonal strength test involves computing the seasonal strength index ($SSI$).

        The $SSI$ ranges between $0$ and $1$, with higher values indicating stronger seasonality in the data. The critical value for the $SSI$ can be obtained from statistical tables based on the sample size and level of significance. If the $SSI$ value exceeds the critical value, the null hypothesis of no seasonality is rejected in favor of the alternative hypothesis of seasonality.

        The seasonal strength test involves the following steps:

        1. Decompose the time series data into its seasonal, trend, and residual components using a method such as seasonal decomposition of time series (STL) or moving average decomposition.
        1. Compute the variance of the seasonal component $Var(S)$ and the variance of the residual component $Var(R)$.
        1. Compute the $SSI$ using the formula above.
        1. Compare the $SSI$ to a critical value from a statistical table for a given significance level and sample size. If the $SSI$ exceeds the critical value, reject the null hypothesis of no seasonality in favor of the alternative hypothesis of seasonality.

        The seasonal strength test is a simple and intuitive test for seasonality in time series data. However, it assumes that the seasonal component is additive and that the residuals are independent and identically distributed. Moreover, it may not be effective for detecting complex seasonal patterns or seasonality in non-stationary or irregular time series data. Therefore, it should be used in conjunction with other tests and techniques for detecting seasonality in time series data.

    Params:
        x (ArrayLike):
            The time series vector.
        m (int):
            The seasonal differencing term. For monthly data, e.g., this would be 12. For quarterly, 4, etc. For the seasonal strength test to work, `m` must exceed 1.

    Returns:
        (float):
            The seasonal strength value.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.algorithms.seasonality import seasonal_strength
        >>> data = load_airline().values
        >>> seasonal_strength(x=data, m=12)
        0.778721...

        ```

    ??? equation "Calculation"

        The $SSI$ is computed using the following formula:

        $$
        SSI = \frac {Var(S)} {Var(S) + Var(R)}
        $$

        where:

        - $Var(S)$ is the variance of the seasonal component, and
        - $Var(R)$ is the variance of the residual component obtained after decomposing the time series data into its seasonal, trend, and residual components using a method such as STL or moving average decomposition.

        ```
        SSI = Var(S) / (Var(S) + Var(R))
        ```

    ??? success "Credit"
        - Inspired by the `tsfeatures` library in both [`Python`](https://github.com/Nixtla/tsfeatures) and [`R`](http://pkg.robjhyndman.com/tsfeatures/).

    ??? question "References"
        - Wang, X, Hyndman, RJ, Smith-Miles, K (2007) "Rule-based forecasting filters using time series features", Computational Statistics and Data Analysis, 52(4), 2244-2259.

    ??? tip "See Also"
        - [`tsfeatures.stl_features`](https://github.com/Nixtla/tsfeatures/blob/main/tsfeatures/tsfeatures.py)
    """
    decomposition = seasonal_decompose(x=x, period=m, model="additive")
    seasonal = np.nanvar(decomposition.seasonal)
    residual = np.nanvar(decomposition.resid)
    return float(seasonal / (seasonal + residual))


@typechecked
def trend_strength(x: ArrayLike, m: int) -> float:
    r"""
    !!! note "Summary"
        The trend strength test is a statistical test for detecting the strength of the trend component in time series data. It measures the extent to which the trend component of a time series explains the variation in the data.

    ???+ abstract "Details"

        The trend strength test involves computing the trend strength index ($TSI$).

        The $TSI$ ranges between $0$ and $1$, with higher values indicating stronger trend in the data. The critical value for the $TSI$ can be obtained from statistical tables based on the sample size and level of significance. If the $TSI$ value exceeds the critical value, the null hypothesis of no trend is rejected in favor of the alternative hypothesis of trend.

        The trend strength test involves the following steps:

        1. Decompose the time series data into its trend, seasonal, and residual components using a method such as seasonal decomposition of time series (STL) or moving average decomposition.
        1. Compute the variance of the trend component, denoted by $Var(T)$.
        1. Compute the variance of the residual component, denoted by $Var(R)$.
        1. Compute the trend strength index ($TSI$) using the formula above.
        1. Compare the $TSI$ value to a critical value based on the sample size and level of significance. If the $TSI$ value exceeds the critical value, reject the null hypothesis of no trend in favor of the alternative hypothesis of trend.

        The trend strength test is a useful tool for identifying the strength of trend in time series data, and it can be used in conjunction with other tests and techniques for detecting trend. However, it assumes that the time series data is stationary and that the trend component is linear. Additionally, it may not be effective for time series data with short time spans or with nonlinear trends. Therefore, it should be used in conjunction with other tests and techniques for detecting trend in time series data.

    Params:
        x (ArrayLike):
            The time series vector.
        m (int):
            The frequency of the time series data set. For the trend strength test to work, `m` must exceed 1.

    Returns:
        (float):
            The trend strength score.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.algorithms.seasonality import trend_strength
        >>> data = load_airline().values
        >>> trend_strength(x=data, m=12)
        0.965679...

        ```

    ??? equation "Calculation"

        The trend strength test involves computing the trend strength index ($TSI$) using the following formula:

        $$
        TSI = \frac{ Var(T) } { Var(T) + Var(R) }
        $$

        where:

        - $Var(T)$ is the variance of the trend component, and
        - $Var(R)$ is the variance of the residual component obtained after decomposing the time series data into its trend, seasonal, and residual components using a method such as STL or moving average decomposition.

        ```
        TSI = Var(T) / (Var(T) + Var(R))
        ```

    ??? success "Credit"
        - Inspired by the `tsfeatures` library in both [`Python`](https://github.com/Nixtla/tsfeatures) and [`R`](http://pkg.robjhyndman.com/tsfeatures/).

    ??? question "References"
        - Wang, X, Hyndman, RJ, Smith-Miles, K (2007) "Rule-based forecasting filters using time series features", Computational Statistics and Data Analysis, 52(4), 2244-2259.

    ??? tip "See Also"
        - [`tsfeatures.stl_features`](https://github.com/Nixtla/tsfeatures/blob/main/tsfeatures/tsfeatures.py)
    """
    decomposition = seasonal_decompose(x=x, period=m, model="additive")
    trend = np.nanvar(decomposition.trend)
    residual = np.nanvar(decomposition.resid)
    return float(trend / (trend + residual))


@typechecked
def spikiness(x: ArrayLike, m: int) -> float:
    r"""
    !!! note "Summary"
        The spikiness test is a statistical test that measures the degree of spikiness or volatility in a time series data. It aims to detect the presence of spikes or sudden changes in the data that may indicate important events or anomalies in the underlying process.

    ???+ abstract "Details"

        The spikiness test involves computing the spikiness index ($SI$). The $SI$ measures the intensity of spikes or outliers in the data relative to the overall variation. A higher $SI$ value indicates a more spiky or volatile time series, while a lower $SI$ value indicates a smoother or less volatile time series.

        The spikiness test involves the following steps:

        1. Decompose the time series data into its seasonal, trend, and residual components using a method such as STL or moving average decomposition.
        1. Compute the mean absolute deviation of the residual component ($MADR$).
        1. Compute the mean absolute deviation of the seasonal component ($MADS$).
        1. Compute the spikiness index ($SI$) using the formula above.

        The spikiness test can be used in conjunction with other tests and techniques for detecting spikes in time series data, such as change point analysis and outlier detection. However, it assumes that the time series data is stationary and that the spikes are abrupt and sudden. Additionally, it may not be effective for time series data with long-term trends or cyclical patterns. Therefore, it should be used in conjunction with other tests and techniques for detecting spikes in time series data.

    Params:
        x (ArrayLike):
            The time series vector.
        m (int):
            The frequency of the time series data set. For the spikiness test to work, `m` must exceed 1.

    Returns:
        (float):
            The spikiness score.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> from ts_stat_tests.utils.data import load_airline
        >>> from ts_stat_tests.algorithms.seasonality import spikiness
        >>> data = load_airline().values
        >>> spikiness(x=data, m=12)
        0.484221...

        ```

    ??? equation "Calculation"

        The spikiness test involves computing the spikiness index ($SI$) using the following formula:

        $$
        SI = \frac {MADR} {MADS}
        $$

        where:

        - $MADR$ is the mean absolute deviation of the residuals, and
        - $MADS$ is the mean absolute deviation of the seasonal component.

        ```
        SI = MADR / MADS
        ```

    ??? success "Credit"
        - All credit to the [`tsfeatures`](http://pkg.robjhyndman.com/tsfeatures/) library. This code is a direct copy+paste from the [`tsfeatures.py`](https://github.com/Nixtla/tsfeatures/blob/master/tsfeatures/tsfeatures.py) module.<br>It is not possible to refer directly to a `spikiness` function in the `tsfeatures` package because the process to calculate seasonal strength is embedded within their `stl_features` function. Therefore, it it necessary to copy it here.

    ??? question "References"
        - Wang, X, Hyndman, RJ, Smith-Miles, K (2007) "Rule-based forecasting filters using time series features", Computational Statistics and Data Analysis, 52(4), 2244-2259.

    ??? tip "See Also"
        - [`tsfeatures.stl_features`](https://github.com/Nixtla/tsfeatures/blob/main/tsfeatures/tsfeatures.py)
    """
    decomposition = seasonal_decompose(x=x, model="additive", period=m)
    madr = np.nanmean(np.abs(decomposition.resid))
    mads = np.nanmean(np.abs(decomposition.seasonal))
    return float(madr / mads)
