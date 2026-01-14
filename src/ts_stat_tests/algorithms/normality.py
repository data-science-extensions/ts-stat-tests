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
from typing import Literal

# ## Python Third Party Imports ----
import numpy as np
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
def jb(x: ArrayLike, axis: int = 0) -> tuple[np.float64, np.float64, np.float64, np.float64]:
    r"""
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

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.normality import jb
        >>> from ts_stat_tests.utils.data import data_airline, data_noise
        >>> airline = data_airline.values
        >>> noise = data_noise

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Using the airline dataset"}
        >>> jb_value, p_value, skew, kurt = jb(airline)
        >>> print(f"{jb_value:.4f}")
        8.9225

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Using random noise"}
        >>> jb_value, p_value, skew, kurt = jb(noise)
        >>> print(f"{jb_value:.4f}")
        0.7478
        >>> print(f"{p_value:.4f}")
        0.6881
        >>> print(f"{skew:.4f}")
        -0.0554
        >>> print(f"{kurt:.4f}")
        3.0753

        ```

    ??? equation "Calculation"
        The Jarque-Bera test statistic is defined as:

        $$
        JB = \frac{n}{6} \left( S^2 + \frac{(K-3)^2}{4} \right)
        $$

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
    return _jb(resids=x, axis=axis)  # type: ignore[return-value]


@typechecked
def ob(x: ArrayLike, axis: int = 0) -> tuple[float, float]:
    r"""
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

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.normality import ob
        >>> from ts_stat_tests.utils.data import data_airline, data_noise
        >>> airline = data_airline.values
        >>> noise = data_noise

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Using the airline dataset"}
        >>> stat, p_val = ob(airline)
        >>> print(f"{stat:.4f}")
        8.6554

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Using random noise"}
        >>> stat, p_val = ob(noise)
        >>> print(f"{stat:.4f}")
        0.8637

        ```

    ??? equation "Calculation"
        The D'Agostino's $K^2$ test statistic is defined as:

        $$
        K^2 = Z_1(g_1)^2 + Z_2(g_2)^2
        $$

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
def sw(x: ArrayLike) -> ShapiroResult:
    r"""
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
        (ShapiroResult):
            A named tuple containing the test statistic and p-value:
            - statistic (float): The test statistic.
            - pvalue (float): The p-value for the hypothesis test.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.normality import sw
        >>> from ts_stat_tests.utils.data import data_airline, data_noise
        >>> airline = data_airline.values
        >>> noise = data_noise

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Using the airline dataset"}
        >>> stat, p_val = sw(airline)
        >>> print(f"{stat:.4f}")
        0.9520

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Using random noise"}
        >>> stat, p_val = sw(noise)
        >>> print(f"{stat:.4f}")
        0.9985

        ```

    ??? equation "Calculation"
        The Shapiro-Wilk test statistic is defined as:

        $$
        W = \frac{\left( \sum_{i=1}^n a_i x_{(i)} \right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}
        $$

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
) -> NormaltestResult:
    r"""
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
        (NormaltestResult):
            A named tuple containing the test statistic and p-value:
            - statistic (float): The test statistic ($K^2$).
            - pvalue (float): A 2-sided chi-squared probability for the hypothesis test.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.normality import dp
        >>> from ts_stat_tests.utils.data import data_airline, data_noise
        >>> airline = data_airline.values
        >>> noise = data_noise

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Using the airline dataset"}
        >>> stat, p_val = dp(airline)
        >>> print(f"{stat:.4f}")
        8.6554

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Using random noise"}
        >>> stat, p_val = dp(noise)
        >>> print(f"{stat:.4f}")
        0.8637

        ```

    ??? equation "Calculation"
        The D'Agostino's $K^2$ test statistic is defined as:

        $$
        K^2 = Z_1(g_1)^2 + Z_2(g_2)^2
        $$

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
    r"""
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
        (AndersonResult):
            A named tuple containing the test statistic, critical values, and significance levels:
            - statistic (float): The Anderson-Darling test statistic.
            - critical_values (list[float]): The critical values for this distribution.
            - significance_level (list[float]): The significance levels for the corresponding critical values in percents.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.algorithms.normality import ad
        >>> from ts_stat_tests.utils.data import data_airline, data_noise
        >>> airline = data_airline.values
        >>> noise = data_noise

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Using the airline dataset"}
        >>> stat, cv, sl = ad(airline)
        >>> print(f"{stat:.4f}")
        1.8185

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Using random normal data"}
        >>> stat, cv, sl = ad(noise)
        >>> print(f"{stat:.4f}")
        0.2325

        ```

    ??? equation "Calculation"
        The Anderson-Darling test statistic $A^2$ is defined as:

        $$
        A^2 = -n - \sum_{i=1}^n \frac{2i-1}{n} \left[ \ln(F(x_i)) + \ln(1 - F(x_{n-i+1})) \right]
        $$

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
