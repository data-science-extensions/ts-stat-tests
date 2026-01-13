# ============================================================================ #
#                                                                              #
#     Title: Normality                                                         #
#     Purpose: Algorithms for testing normality of data.                       #
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
    This module provides implementations of various statistical tests to assess the normality of data distributions. These tests are essential in statistical analysis and time series forecasting, as many models assume that the underlying data follows a normal distribution.
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
from typing import Literal, Union

# ## Python Third Party Imports ----
from numpy.typing import ArrayLike
from scipy.stats import anderson as _ad, normaltest as _dp, shapiro as _sw
from scipy.stats._morestats import AndersonResult, ShapiroResult
from scipy.stats._stats_py import NormaltestResult
from statsmodels.stats.stattools import jarque_bera as _jb, omni_normtest as _ob
from typeguard import typechecked


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


@typechecked
def jb(
    x: ArrayLike,
    axis: int = 0,
) -> tuple[
    Union[float, ArrayLike],
    Union[float, ArrayLike],
    Union[float, ArrayLike],
    Union[float, ArrayLike],
]:
    """
    !!! note "Summary"
        The Jarque-Bera test is a statistical test used to determine whether a dataset follows a normal distribution. In time series forecasting, the test can be used to evaluate whether the residuals of a model follow a normal distribution.

    ???+ abstract "Details"
        To apply the Jarque-Bera test to time series data, we first need to estimate the residuals of the forecasting model. The residuals represent the difference between the actual values of the time series and the values predicted by the model. We can then use the Jarque-Bera test to evaluate whether the residuals follow a normal distribution.

        The Jarque-Bera test is based on two statistics, skewness and kurtosis, which measure the degree of asymmetry and peakedness in the distribution of the residuals. The test compares the observed skewness and kurtosis of the residuals to the expected values for a normal distribution. If the observed values are significantly different from the expected values, the test rejects the null hypothesis that the residuals follow a normal distribution.

    Params:
        x (ArrayLike):
            Data to test for normality. Usually regression model residuals that are mean 0.
        axis (int):
            Axis to use if data has more than 1 dimension.
            Default: `0`

    Raises:
        (ValueError):
            If the input data `x` is invalid.

    Returns:
        JB (float):
            The Jarque-Bera test statistic.
        JBpv (float):
            The pvalue of the test statistic.
        skew (float):
            Estimated skewness of the data.
        kurtosis (float):
            Estimated kurtosis of the data.

    ???+ example "Examples"

        Example one, using the `airline` data from the `sktime` package.

        ```pycon {.py .python linenums="1" title="Python"}
        >>> # Import packages
        >>> from sktime.datasets import load_airline
        >>> from statsmodels.stats.stattools import jarque_bera

        >>> # Load the airline dataset
        >>> y = load_airline()

        >>> # Apply Jarque-Bera test
        >>> jb_value, p_value, skewness, kurtosis = jarque_bera(y)

        >>> # Print the results
        >>> print("Jarque-Bera test statistic:", jb_value)
        Jarque-Bera test statistic: 4.588031669436549
        >>> print("p-value:", p_value)
        p-value: 0.10134805179561781

        >>> # Check the test
        >>> if p_value < 0.05:
        ...     print("Reject the null hypothesis that the data is normally distributed")
        ... else:
        ...     print("Cannot reject the null hypothesis that the data is normally distributed")
        ...
        Cannot reject the null hypothesis that the data is normally distributed
        ```

        In this example, the p-value is **greater** than `0.05`, indicating that we _cannot_ reject the null hypothesis that the data _is_ normally distributed. Therefore, we can assume that the airline data **does** follow a normal distribution.

        ---

        Example two, using the `sine` wave data generated from the `numpy` package.

        ```pycon {.py .python linenums="1" title="Python"}
        >>> # Import packages
        >>> import numpy as np
        >>> from statsmodels.stats.stattools import jarque_bera

        >>> # Generate sine wave data
        >>> t = np.linspace(0, 10, 100)
        >>> y = np.sin(t)

        >>> # Apply Jarque-Bera test
        >>> jb_value, p_value, skewness, kurtosis = jarque_bera(y)

        >>> # Print the results
        >>> print("Jarque-Bera test statistic:", jb_value)
        Jarque-Bera test statistic: 15.830310292715973
        >>> print("p-value:", p_value)
        p-value: 0.00036833142556487206

        >>> if p_value < 0.05:
        ...     print("Reject the null hypothesis that the data is normally distributed")
        ... else:
        ...     print("Cannot reject the null hypothesis that the data is normally distributed")
        ...
        Reject the null hypothesis that the data is normally distributed
        ```

        In this example, the p-value is **less** than `0.05`, indicating that we _can_ reject the null hypothesis that the data _is_ normally distributed. Therefore, we can assume that the sine wave data does **not** follow a normal distribution.

        ---

        Example three, using the `FractionalGaussianNoise` random data generated from the `stochastic` package.

    ??? equation "Calculation"
        The Jarque-Bera test statistic is defined as:

        $$JB = \frac{n}{6} \left( S^2 + \frac{(K-3)^2}{4} \right)$$

        where:

        - $n$ is the sample size,
        - $S$ is the sample skewness, and
        - $K$ is the sample kurtosis.

    ??? note "Notes"
        Each output returned has 1 dimension fewer than data.
        The Jarque-Bera test statistic tests the null that the data is normally distributed against an alternative that the data follow some other distribution. It has an asymptotic $\chi_2^2$ distribution.

    ??? success "Credit"
        All credit goes to the [`statsmodels`](https://www.statsmodels.org) library.

    ??? question "References"
        - Jarque, C. and Bera, A. (1980) "Efficient tests for normality, homoscedasticity and serial independence of regression residuals", 6 Econometric Letters 255-259.

    ??? tip "See Also"
        - [`ob()`][ts_stat_tests.algorithms.normality.ob]
        - [`sw()`][ts_stat_tests.algorithms.normality.sw]
        - [`dp()`][ts_stat_tests.algorithms.normality.dp]
        - [`ad()`][ts_stat_tests.algorithms.normality.ad]
    """
    return _jb(resids=x, axis=axis)


@typechecked
def ob(
    x: ArrayLike,
    axis: int = 0,
) -> Union[tuple[float, float], NormaltestResult]:
    """
    !!! note "Summary"
        The Omnibus test is a statistical test used to evaluate the normality of a dataset, including time series data. In time series forecasting, the Omnibus test can be used to assess whether the residuals of a model follow a normal distribution.

    ???+ abstract "Details"
        The Omnibus test uses a combination of skewness and kurtosis measures to assess whether the residuals follow a normal distribution. Skewness measures the degree of asymmetry in the distribution of the residuals, while kurtosis measures the degree of peakedness or flatness. If the residuals follow a normal distribution, their skewness and kurtosis should be close to zero.

    Params:
        x (ArrayLike):
            Data to test for normality. Usually regression model residuals that are mean 0.
        axis (int):
            Axis to use if data has more than 1 dimension.
            Default: `0`

    Raises:
        (ValueError):
            If the input data `x` is invalid.

    Returns:
        statistic (float):
            The Omnibus test statistic.
        pvalue (float):
            The p-value for the hypothesis test.

    ???+ example "Examples"
        Example one, using the `airline` data from the `sktime` package.

        ```pycon {.py .python linenums="1" title="Python"}
        >>> # Import packages
        >>> from sktime.datasets import load_airline
        >>> from statsmodels.stats.stattools import omni_normtest

        >>> # load the airline dataset
        >>> airline = load_airline()

        >>> # run the Omnibus test on the dataset
        >>> statistic, p_value = omni_normtest(airline)

        >>> # print the results
        >>> print(f"Omnibus test statistic: {statistic:.3f}")
        Omnibus test statistic: 1.753
        >>> print(f"Omnibus test p-value: {p_value:.3f}")
        Omnibus test p-value: 0.416

        >>> # check if null hypothesis is rejected
        >>> alpha = 0.05
        >>> if p_value < alpha:
        ...     print("Reject null hypothesis that data is normally distributed")
        ... else:
        ...     print("Fail to reject null hypothesis that data is normally distributed")
        ...
        Fail to reject null hypothesis that data is normally distributed
        ```

        The null hypothesis of the Omnibus test is that the data _is_ normally distributed. In this case, the p-value is `0.416`, which is **greater** than the significance level of `0.05`, indicating that we _fail_ to reject the null hypothesis. Therefore, we can conclude that the Airline dataset **is** likely normally distributed.

        ---

        Example two, using the `sine` wave data generated from the `numpy` package.


    ??? equation "Calculation"
        The D'Agostino's $K^2$ test statistic is defined as:

        $$K^2 = Z_1(g_1)^2 + Z_2(g_2)^2$$

        where:

        - $Z_1(g_1)$ is the standard normal transformation of skewness, and
        - $Z_2(g_2)$ is the standard normal transformation of kurtosis.

    ??? note "Notes"
        The Omnibus test statistic tests the null that the data is normally distributed against an alternative that the data follow some other distribution. It is based on D'Agostino's $K^2$ test statistic.

    ??? success "Credit"
        All credit goes to the [`statsmodels`](https://www.statsmodels.org) library.

    ??? question "References"
        - D'Agostino, R. B. and Pearson, E. S. (1973), "Tests for departure from normality," Biometrika, 60, 613-622.
        - D'Agostino, R. B. and Stephens, M. A. (1986), "Goodness-of-fit techniques," New York: Marcel Dekker.

    ??? tip "See Also"
        - [`jb()`][ts_stat_tests.algorithms.normality.jb]
        - [`sw()`][ts_stat_tests.algorithms.normality.sw]
        - [`dp()`][ts_stat_tests.algorithms.normality.dp]
        - [`ad()`][ts_stat_tests.algorithms.normality.ad]
    """
    return _ob(resids=x, axis=axis)


@typechecked
def sw(
    x: ArrayLike,
) -> Union[tuple[float, float], ShapiroResult]:
    """
    !!! note "Summary"
        The Shapiro-Wilk test is a statistical test used to determine whether a dataset follows a normal distribution.

    ???+ abstract "Details"
        The Shapiro-Wilk test is based on the null hypothesis that the residuals of the forecasting model are normally distributed. The test calculates a test statistic that compares the observed distribution of the residuals to the expected distribution under the null hypothesis of normality.

    Params:
        x (ArrayLike):
            Array of sample data.

    Raises:
        (ValueError):
            If the input data `x` is invalid.

    Returns:
        statistic (float):
            The test statistic.
        pvalue (float):
            The p-value for the hypothesis test.

    ???+ example "Examples"
        Test the null hypothesis that a random sample was drawn from a normal distribution.

        ```pycon {.py .python linenums="1" title="From the `scipy` docs"}
        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> x = stats.norm.rvs(loc=5, scale=3, size=100, random_state=rng)
        >>> shapiro_test = stats.shapiro(x)
        >>> shapiro_test
        ShapiroResult(statistic=0.9813305735588074, pvalue=0.16855233907699585)
        >>> shapiro_test.statistic
        0.9813305735588074
        >>> shapiro_test.pvalue
        0.16855233907699585
        ```

        ---

        Example one, using the `airline` data from the `sktime` package.

        ```pycon {.py .python linenums="1" title="Python"}
        >>> # Import packages
        >>> from sktime.datasets import load_airline
        >>> from scipy.stats import shapiro

        >>> # load the airline data
        >>> data = load_airline()

        >>> # run the Shapiro-Wilk test on the data
        >>> statistic, p_value = shapiro(data)

        >>> # print the results
        >>> print(f"Shapiro-Wilk test statistic: {statistic:.3f}")
        Shapiro-Wilk test statistic: 0.910
        >>> print(f"Shapiro-Wilk test p-value: {p_value:.3f}")
        Shapiro-Wilk test p-value: 0.054

        >>> # check if null hypothesis is rejected
        >>> alpha = 0.05
        >>> if p_value < alpha:
        ...     print("Reject null hypothesis that data is normally distributed")
        ... else:
        ...     print("Fail to reject null hypothesis that data is normally distributed")
        ...
        Fail to reject null hypothesis that data is normally distributed
        ```

        The null hypothesis of the Shapiro-Wilk test is that the data _is_ normally distributed. In this case, the p-value is `0.054`, which is **greater** than the significance level of `0.05`, indicating that we _fail_ to reject the null hypothesis. Therefore, we can conclude that the airline data **is** likely normally distributed.

        ---

        Example two, using the `sine` wave data generated from the `numpy` package.


    ??? equation "Calculation"
        The Shapiro-Wilk test statistic is defined as:

        $$W = \frac{\left( \sum_{i=1}^n a_i x_{(i)} \right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}$$

        where:

        - $x_{(i)}$ are the ordered sample values,
        - $\bar{x}$ is the sample mean, and
        - $a_i$ are constants generated from the covariances, variances and means of the order statistics of a sample of size $n$ from a normal distribution.

    ??? note "Notes"
        The algorithm used is described in (Algorithm as R94 Appl. Statist. (1995)) but censoring parameters as described are not implemented. For $N > 5000$ the $W$ test statistic is accurate but the $p-value$ may not be.

    ??? success "Credit"
        All credit goes to the [`scipy`](https://docs.scipy.org/) library.

    ??? question "References"
        - Shapiro, S. S. & Wilk, M.B (1965). An analysis of variance test for normality (complete samples), Biometrika, Vol. 52, pp. 591-611.
        - Algorithm as R94 Appl. Statist. (1995) VOL. 44, NO. 4.

    ??? tip "See Also"
        - [`jb()`][ts_stat_tests.algorithms.normality.jb]
        - [`ob()`][ts_stat_tests.algorithms.normality.ob]
        - [`dp()`][ts_stat_tests.algorithms.normality.dp]
        - [`ad()`][ts_stat_tests.algorithms.normality.ad]
    """
    return _sw(x=x)


@typechecked
def dp(
    x: ArrayLike,
    axis: int = 0,
    nan_policy: VALID_DP_NAN_POLICY_OPTIONS = "propagate",
) -> Union[
    tuple[Union[float, ArrayLike], Union[float, ArrayLike]],
    NormaltestResult,
]:
    """
    !!! note "Summary"
        The D'Agostino and Pearson's test is a statistical test used to evaluate whether a dataset follows a normal distribution.

    ???+ abstract "Details"
        The D'Agostino and Pearson's test uses a combination of skewness and kurtosis measures to assess whether the residuals follow a normal distribution. Skewness measures the degree of asymmetry in the distribution of the residuals, while kurtosis measures the degree of peakedness or flatness.

    Params:
        x (ArrayLike):
            The array containing the sample to be tested.
        axis (int):
            Axis along which to compute test. If `None`, compute over the whole array `a`.
            Default: `0`
        nan_policy (VALID_DP_NAN_POLICY_OPTIONS):
            Defines how to handle when input contains nan.
            
            - `"propagate"`: returns nan
            - `"raise"`: throws an error
            - `"omit"`: performs the calculations ignoring nan values
            
            Default: `"propagate"`

    Raises:
        (ValueError):
            If the input data `x` is invalid.

    Returns:
        statistic (float):
            The test statistic ($K^2$).
        pvalue (float):
            A 2-sided chi-squared probability for the hypothesis test.

    ???+ example "Examples"

        Test the null hypothesis that a random sample was drawn from a normal distribution.

        ```pycon {.py .python linenums="1" title="From the `scipy` docs"}
        >>> import numpy as np
        >>> from scipy import stats
        >>> rng = np.random.default_rng()
        >>> pts = 1000
        >>> a = rng.normal(0, 1, size=pts)
        >>> b = rng.normal(2, 1, size=pts)
        >>> x = np.concatenate((a, b))
        >>> k2, p = stats.normaltest(x)
        >>> alpha = 1e-3
        >>> print("p = {:g}".format(p))
        p = 8.4713e-19
        >>> if p < alpha:  # null hypothesis: x comes from a normal distribution
        ...     print("The null hypothesis can be rejected")
        ... else:
        ...     print("The null hypothesis cannot be rejected")
        ...
        "The null hypothesis can be rejected"
        ```

        ---

        Example one, using the `airline` data from the `sktime` package.

        ```pycon {.py .python linenums="1" title="Python"}
        >>> # Import packages
        >>> from sktime.datasets import load_airline
        >>> from scipy.stats import normaltest

        >>> # load the airline data
        >>> data = load_airline()

        >>> # run D'Agostino and Pearson's test on the data
        >>> statistic, p_value = normaltest(data)

        >>> # print the results
        >>> print(f"D'Agostino and Pearson's test statistic: {statistic:.3f}")
        D'Agostino and Pearson's test statistic: 7.764
        >>> print(f"D'Agostino and Pearson's test p-value: {p_value:.3f}")
        D'Agostino and Pearson's test p-value: 0.021

        >>> # check if null hypothesis is rejected
        >>> alpha = 0.05
        >>> if p_value < alpha:
        ...     print("Reject null hypothesis that data is normally distributed")
        ... else:
        ...     print("Fail to reject null hypothesis that data is normally distributed")
        ...
        Reject null hypothesis that data is normally distributed
        ```

        The null hypothesis of D'Agostino and Pearson's test is that the data _is_ normally distributed. In this case, the p-value is `0.021`, which is **less** than the significance level of `0.05`, indicating that we _can_ reject the null hypothesis. Therefore, we can conclude that the airline data from the sktime library is **not** normally distributed.

        ---

        Example two, using the `sine` wave data generated from the `numpy` package.

        ```pycon {.py .python linenums="1" title="Python"}
        >>> # Import packages
        >>> import numpy as np
        >>> from scipy.stats import normaltest

        >>> # generate sine wave data
        >>> data = np.sin(np.linspace(0, 2 * np.pi, num=100))

        >>> # run D'Agostino and Pearson's test on the data
        >>> statistic, p_value = normaltest(data)

        >>> # print the results
        >>> print(f"D'Agostino and Pearson's test statistic: {statistic:.3f}")
        D'Agostino and Pearson's test statistic: 50.583
        >>> print(f"D'Agostino and Pearson's test p-value: {p_value:.3f}")
        D'Agostino and Pearson's test p-value: 0.000

        >>> # check if null hypothesis is rejected
        >>> alpha = 0.05
        >>> if p_value < alpha:
        ...     print("Reject null hypothesis that data is normally distributed")
        ... else:
        ...     print("Fail to reject null hypothesis that data is normally distributed")
        ...
        Reject null hypothesis that data is normally distributed
        ```

        The null hypothesis of D'Agostino and Pearson's test is that the data _is_ normally distributed. In this case, the p-value is `0.000`, which is **less** than the significance level of `0.05`, indicating that we _can_ reject the null hypothesis. Therefore, we can conclude that the sine wave data generated from the numpy library is **not** normally distributed.

        ---

        Example three, using the `FractionalGaussianNoise` random data generated from the `stochastic` package.


    ??? equation "Calculation"
        The D'Agostino's $K^2$ test statistic is defined as:

        $$K^2 = Z_1(g_1)^2 + Z_2(g_2)^2$$

        where:

        - $Z_1(g_1)$ is the standard normal transformation of skewness, and
        - $Z_2(g_2)$ is the standard normal transformation of kurtosis.

    ??? note "Notes"
        This function is a wrapper for the `scipy.stats.normaltest` function.

    ??? success "Credit"
        All credit goes to the [`scipy`](https://docs.scipy.org/) library.

    ??? question "References"
        - D'Agostino, R. B. (1971), "An omnibus test of normality for moderate and large sample size", Biometrika, 58, 341-348
        - D'Agostino, R. and Pearson, E. S. (1973), "Tests for departure from normality", Biometrika, 60, 613-622

    ??? tip "See Also"
        - [`jb()`][ts_stat_tests.algorithms.normality.jb]
        - [`ob()`][ts_stat_tests.algorithms.normality.ob]
        - [`sw()`][ts_stat_tests.algorithms.normality.sw]
        - [`ad()`][ts_stat_tests.algorithms.normality.ad]
    """
    return _dp(a=x, axis=axis, nan_policy=nan_policy)


@typechecked
def ad(
    x: ArrayLike,
    dist: VALID_AD_DIST_OPTIONS = "norm",
) -> AndersonResult:
    """
    !!! note "Summary"
        The Anderson-Darling test is a statistical test used to evaluate whether a dataset follows a normal distribution.

    ???+ abstract "Details"
        The Anderson-Darling test tests the null hypothesis that a sample is drawn from a population that follows a particular distribution. For the Anderson-Darling test, the critical values depend on which distribution is being tested against.

    Params:
        x (ArrayLike):
            Array of sample data.
        dist (VALID_AD_DIST_OPTIONS):
            The type of distribution to test against.
            Default: `"norm"`

    Raises:
        (ValueError):
            If the input data `x` is invalid.

    Returns:
        statistic (float):
            The Anderson-Darling test statistic.
        critical_values (list[float]):
            The critical values for this distribution.
        significance_level (list[float]):
            The significance levels for the corresponding critical values in percents.

    ???+ example "Examples"

        Test the null hypothesis that a random sample was drawn from a normal distribution (with unspecified mean and standard deviation).

        ```pycon {.py .python linenums="1" title="From the `scipy` docs"}
        >>> import numpy as np
        >>> from scipy.stats import anderson
        >>> rng = np.random.default_rng()
        >>> data = rng.random(size=35)
        >>> res = anderson(data)
        >>> res.statistic
        0.8398018749744764
        >>> res.critical_values
        array([0.527, 0.6  , 0.719, 0.839, 0.998])
        >>> res.significance_level
        array([15. , 10. ,  5. ,  2.5,  1. ])
        ```

        The value of the statistic (barely) exceeds the critical value associated with a significance level of $2.5%$, so the null hypothesis may be rejected at a significance level of $2.5%$, but not at a significance level of $1%$.

        ---

        Example one, using the `airline` data from the `sktime` package.

        ```pycon {.py .python linenums="1" title="Python"}
        >>> # Import packages
        >>> from sktime.datasets import load_airline
        >>> from scipy.stats import anderson

        >>> # load the airline data
        >>> airline_data = load_airline()

        >>> # run Anderson-Darling test on the data
        >>> result = anderson(airline_data)

        >>> # print the results
        >>> print(f"Anderson-Darling test statistic: {result.statistic:.3f}")
        Anderson-Darling test statistic: 3.089
        >>> print(f"Anderson-Darling test critical values: {result.critical_values}")
        Anderson-Darling test critical values: [0.565 0.644 0.772 0.901 1.072]
        >>> print(f"Anderson-Darling test significance levels: {result.significance_level}")
        Anderson-Darling test significance levels: [15.  10.   5.   2.5  1. ]

        >>> # check if null hypothesis is rejected
        >>> alpha = 0.05
        >>> if result.statistic > result.critical_values[2]:
        ...     print("Reject null hypothesis that data is normally distributed")
        ... else:
        ...     print("Fail to reject null hypothesis that data is normally distributed")
        ...
        Reject null hypothesis that data is normally distributed
        ```

        The null hypothesis of Anderson-Darling test is that the data is normally distributed. In this case, the test statistic is 3.089, which is greater than the critical value at 5% significance level of 0.772, indicating that we reject the null hypothesis. Therefore, we can conclude that the airline data from the sktime library is not normally distributed.

        ---

        Example two, using the `sine` wave data generated from the `numpy` package.


    ??? equation "Calculation"
        The Anderson-Darling test statistic $A^2$ is defined as:

        $$A^2 = -n - \sum_{i=1}^n \frac{2i-1}{n} \left[ \ln(F(x_i)) + \ln(1 - F(x_{n-i+1})) \right]$$

        where:

        - $n$ is the sample size,
        - $F$ is the cumulative distribution function of the specified distribution, and
        - $x_i$ are the ordered sample values.

    ??? note "Notes"
        Critical values provided are for the following significance levels:
        - normal/exponential: 15%, 10%, 5%, 2.5%, 1%
        - logistic: 25%, 10%, 5%, 2.5%, 1%, 0.5%
        - Gumbel: 25%, 10%, 5%, 2.5%, 1%

    ??? success "Credit"
        All credit goes to the [`scipy`](https://docs.scipy.org/) library.

    ??? question "References"
        - Stephens, M. A. (1974). EDF Statistics for Goodness of Fit and Some Comparisons, Journal of the American Statistical Association, Vol. 69, pp. 730-737.

    ??? tip "See Also"
        - [`jb()`][ts_stat_tests.algorithms.normality.jb]
        - [`ob()`][ts_stat_tests.algorithms.normality.ob]
        - [`sw()`][ts_stat_tests.algorithms.normality.sw]
        - [`dp()`][ts_stat_tests.algorithms.normality.dp]
    """
    return _ad(x=x, dist=dist)
