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
        The Jarque-Bera test is a statistical test used to determine whether a dataset follows a normal distribution. In time series forecasting, the test can be used to evaluate whether the residuals of a model follow a normal distribution, which is an assumption of many time series forecasting models.

    ???+ abstract "Details"
        To apply the Jarque-Bera test to time series data, we first need to estimate the residuals of the forecasting model. The residuals represent the difference between the actual values of the time series and the values predicted by the model. We can then use the Jarque-Bera test to evaluate whether the residuals follow a normal distribution.

        The Jarque-Bera test is based on two statistics, skewness and kurtosis, which measure the degree of asymmetry and peakedness in the distribution of the residuals. The test compares the observed skewness and kurtosis of the residuals to the expected values for a normal distribution. If the observed values are significantly different from the expected values, the test rejects the null hypothesis that the residuals follow a normal distribution.

        In practice, we can use statistical software to perform the Jarque-Bera test on the residuals of a time series forecasting model. If the test indicates that the residuals do not follow a normal distribution, we may need to consider modifying the forecasting model or using a different modeling approach.

    Params:
        x (ArrayLike):
            Data to test for normality. Usually regression model residuals that are mean 0.
        axis (int):
            Axis to use if data has more than 1 dimension.<br>
            Defaults to `0`.

    Returns:
        JB (Union[float, ArrayLike]):
            The Jarque-Bera test statistic.
        JBpv (Union[float, ArrayLike]):
            The pvalue of the test statistic.
        skew (Union[float, ArrayLike]):
            Estimated skewness of the data.
        kurtosis (Union[float, ArrayLike]):
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

        ```pycon {.py .python linenums="1" title="Python"}
        >>> # Import packages
        >>> from stochastic.noise import FractionalGaussianNoise
        >>> from statsmodels.stats.stattools import jarque_bera

        >>> # Generate Fractional Gaussian Noise
        >>> fgn = FractionalGaussianNoise(t=1, hurst=0.5, length=100, method="daviesharte")
        >>> noise = fgn.sample()

        >>> # Apply Jarque-Bera test
        >>> jb_value, p_value, skewness, kurtosis = jarque_bera(y)

        >>> # Print the results
        >>> print("Jarque-Bera test statistic:", jb_value)
        Jarque-Bera test statistic: 8.94626982252318
        >>> print("p-value:", p_value)
        p-value: 0.011411891515478784

        >>> if p_value < 0.05:
        ...     print("Reject the null hypothesis that the data is normally distributed")
        ... else:
        ...     print("Cannot reject the null hypothesis that the data is normally distributed")
        ...
        Reject the null hypothesis that the data is normally distributed
        ```

        In this example, the p-value is **less** than `0.05`, indicating that we _can_ reject the null hypothesis that the data _is_ normally distributed. Therefore, we can assume that the random noise generated by `FractionalGaussianNoise` does **not** follow a normal distribution.

    ??? note "Notes"
        Each output returned has 1 dimension fewer than data.

        The Jarque-Bera test statistic tests the null that the data is normally distributed against an alternative that the data follow some other distribution. The test statistic is based on two moments of the data, the skewness, and the kurtosis, and has an asymptotic $x_2^2$ distribution.

        The test statistic is defined as:

        $$
        JB = n \\left( \\frac{S^2}{6} + \\frac{(K-3)^2}{24} \\right)
        $$

        where:

        - $n$ is the number of data points,
        - $S$ is the sample skewness, and
        - $K$ is the sample kurtosis of the data.

    ??? success "Credit"
        - All credit goes to the [`statsmodels`](https://www.statsmodels.org) library.

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
        The Omnibus test is a statistical test used to evaluate the normality of a dataset, including time series data. In time series forecasting, the Omnibus test can be used to assess whether the residuals of a model follow a normal distribution, which is an important assumption for many statistical models.

    ???+ abstract "Details"
        The Omnibus test uses a combination of skewness and kurtosis measures to assess whether the residuals follow a normal distribution. Skewness measures the degree of asymmetry in the distribution of the residuals, while kurtosis measures the degree of peakedness or flatness. If the residuals follow a normal distribution, their skewness and kurtosis should be close to zero.

        To apply the Omnibus test to time series data, we first need to estimate the residuals of the forecasting model. We can then use a statistical software package to perform the Omnibus test on the residuals. The test produces a single p-value, which indicates the probability of observing the observed skewness and kurtosis values if the residuals follow a normal distribution. If the p-value is greater than the significance level (usually 0.05), we can conclude that the residuals follow a normal distribution.

        If the Omnibus test indicates that the residuals do not follow a normal distribution, we may need to consider using a different modeling approach or modifying the forecasting model. It is important to ensure that the residuals of a time series forecasting model follow a normal distribution to ensure that the model is valid and reliable for making predictions.

        The Omnibus test for normality is a statistical test used to evaluate whether a dataset, including time series data, follows a normal distribution. The mathematical equation for the Omnibus test is:

        $$
        O = N \\times (b_1^2 + b_2^2)
        $$

        where:

        - $O$ is the Omnibus test statistic
        - $N$ is the sample size
        - $b1$ and $b2$ are the coefficients of the first two terms of a third-order polynomial fit to the data

        To calculate the Omnibus test statistic for time series data, we need to perform the following steps:

        1. Estimate the residuals of the forecasting model: The residuals are the difference between the actual values and the predicted values of the time series model.

        1. Calculate the sample mean and standard deviation of the residuals: These are the mean and standard deviation of the residuals, respectively.

        1. Calculate the skewness and kurtosis of the residuals: These are measures of the asymmetry and peakedness of the distribution of the residuals, respectively.

        1. Fit a third-order polynomial to the standardized residuals: The standardized residuals are the residuals divided by their sample standard deviation. The third-order polynomial has the form:

            $$
            z = b_0 + (b_1 \\times x) + (b_2 \\times x^2) + (b_3 \\times x^3)
            $$

            where:

            - $z$ is the standardized residual,
            - $x$ is the normal deviate (i.e., the value that would be expected if the data followed a normal distribution), and
            - $b_0$, $b_1$, $b_2$, and $b_3$ are the coefficients of the polynomial fit.

        1. Calculate the values of $b_1$ and $b_2$: These are the coefficients of the first two terms of the polynomial fit.

        1. Substitute the values for sample size, $b_1$, and $b_2$ into the Omnibus formula: The formula calculates a single test statistic, which is the Omnibus value.

        1. Compare the Omnibus value to a critical value from a chi-squared distribution with 2 degrees of freedom: If the Omnibus value is greater than the critical value, we can reject the null hypothesis of normality and conclude that the residuals do not follow a normal distribution. If the Omnibus value is less than the critical value, we cannot reject the null hypothesis of normality and can conclude that the residuals follow a normal distribution.

        In summary, the Omnibus test is a statistical test that evaluates normality of time series residuals using a third-order polynomial fit to the standardized residuals. It calculates a single test statistic using the coefficients of the polynomial fit and compares it to a critical value to determine whether the residuals follow a normal distribution.

    Params:
        x (ArrayLike):
            Data to test for normality. Usually regression model residuals that are mean 0.
        axis (int):
            Axis to use if data has more than 1 dimension.<br>
            Defaults to `0`.

    Returns:
        statistic (Union[float, np.ndarray]):
            The Omnibus test statistic.
        pvalue (Union[float, np.ndarray]):
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

        ```pycon {.py .python linenums="1" title="Python"}
        >>> # Import packages
        >>> import numpy as np
        >>> from statsmodels.stats.stattools import omni_normtest

        >>> # generate sine wave data
        >>> data = np.sin(np.linspace(0, 2 * np.pi, num=100))

        >>> # run the Omnibus test on the data
        >>> statistic, p_value = omni_normtest(data)

        >>> # print the results
        >>> print(f"Omnibus test statistic: {statistic:.3f}")
        Omnibus test statistic: 24.750
        >>> print(f"Omnibus test p-value: {p_value:.3f}")
        Omnibus test p-value: 4.326e-06

        >>> # check if null hypothesis is rejected
        >>> alpha = 0.05
        >>> if p_value < alpha:
        ...     print("Reject null hypothesis that data is normally distributed")
        ... else:
        ...     print("Fail to reject null hypothesis that data is normally distributed")
        ...
        Reject null hypothesis that data is normally distributed
        ```

        The null hypothesis of the Omnibus test _is_ that the data is normally distributed. In this case, the p-value is `4.326e-06`, which is much **smaller** than the significance level of `0.05`, indicating strong evidence to _reject_ the null hypothesis. Therefore, we can conclude that the sine wave data is **not** normally distributed.

        ---

        Example three, using the `FractionalGaussianNoise` random data generated from the `stochastic` package.

        ```pycon {.py .python linenums="1" title="Python"}
        >>> # Import packages
        >>> from stochastic.noise import FractionalGaussianNoise
        >>> from statsmodels.stats.stattools import omni_normtest

        >>> # Generate Fractional Gaussian Noise
        >>> fgn = FractionalGaussianNoise(t=1, hurst=0.5, length=1000, method="daviesharte")
        >>> noise = fgn.sample()

        >>> # run the Omnibus test on the data
        >>> statistic, p_value = omni_normtest(noise)

        >>> # print the results
        >>> print(f"Omnibus test statistic: {statistic:.3f}")
        Omnibus test statistic: 4.717
        >>> print(f"Omnibus test p-value: {p_value:.3f}")
        Omnibus test p-value: 0.094

        >>> # check if null hypothesis is rejected
        >>> alpha = 0.05
        >>> if p_value < alpha:
        ...     print("Reject null hypothesis that data is normally distributed")
        ... else:
        ...     print("Fail to reject null hypothesis that data is normally distributed")
        ...
        Fail to reject null hypothesis that data is normally distributed
        ```

        The null hypothesis of the Omnibus test _is_ that the data is normally distributed. In this case, the p-value is `0.094`, which is **greater** than the significance level of `0.05`, indicating that we _fail_ to reject the null hypothesis. Therefore, we can conclude that the random noise generated by the `FractionalGaussianNoise` class **is** likely normally distributed.

    ??? note "Notes"
        The Omnibus test statistic tests the null that the data is normally distributed against an alternative that the data follow some other distribution. It is based on D'Agostino's $K^2$ test statistic.

    ??? success "Credit"
        - All credit goes to the [`statsmodels`](https://www.statsmodels.org) library.

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
        The Shapiro-Wilk test is a statistical test used to determine whether a dataset, including time series data, follows a normal distribution. In time series forecasting, the Shapiro-Wilk test can be used to evaluate whether the residuals of a model follow a normal distribution, which is an assumption of many statistical models.

    ???+ abstract "Details"
        The Shapiro-Wilk test is based on the null hypothesis that the residuals of the forecasting model are normally distributed. The test calculates a test statistic that compares the observed distribution of the residuals to the expected distribution under the null hypothesis of normality. If the observed distribution of the residuals deviates significantly from the expected distribution under normality, the test rejects the null hypothesis and concludes that the residuals do not follow a normal distribution.

        To apply the Shapiro-Wilk test to time series data, we first need to estimate the residuals of the forecasting model. We can then use a statistical software package to perform the Shapiro-Wilk test on the residuals. The test produces a p-value, which indicates the probability of observing the observed distribution of the residuals if the null hypothesis of normality is true. If the p-value is less than the significance level (usually 0.05), we can conclude that the residuals do not follow a normal distribution.

        If the Shapiro-Wilk test indicates that the residuals do not follow a normal distribution, we may need to consider using a different modeling approach or modifying the forecasting model. It is important to ensure that the residuals of a time series forecasting model follow a normal distribution to ensure that the model is valid and reliable for making predictions.

        The Shapiro-Wilk test is a statistical test used to evaluate whether a dataset, including time series data, follows a normal distribution. The mathematical equation for the Shapiro-Wilk test is:

        $$
        W = \\frac { \\left( \\sum_{i=1}^n (a_i \\times z_i) \\right)^2 } { \\sum_{i=1}^n (x_i - \\bar{x})^2 }
        $$

        where:

        - $W$ is the test statistic
        - $a_i$ are the coefficients calculated from the ordered sample values
        - $z_i$ are the corresponding normal deviates for the ai coefficients
        - $x_i$ are the ordered sample values
        - $\\bar{x}$ is the sample mean

        To calculate the Shapiro-Wilk test statistic for time series data, we need to perform the following steps:

        1. Estimate the residuals of the forecasting model: The residuals are the difference between the actual values and the predicted values of the time series model.

        1. Calculate the sample mean and standard deviation of the residuals: These are the mean and standard deviation of the residuals, respectively.

        1. Standardize the residuals: The standardized residuals are the residuals divided by their sample standard deviation.

        1. Order the standardized residuals from smallest to largest: This step ensures that the Shapiro-Wilk test is performed on a sample that is in ascending order.

        1. Calculate the $a_i$ coefficients: The $a_i$ coefficients are calculated from the ordered sample values using the formula:

            $$
            a_i = \\frac { \\sum_{j=1}^{n} (a_{i_j} \\times x_j) }{ s^2 }
            $$

            where:

            - $s^2$ is the sample variance and
            - $a_{i_j}$ are constants that depend on the sample size and the order of the sample values. These constants are pre-calculated and available in statistical software packages.

        1. Calculate the $z_i$ normal deviates: The $z_i$ normal deviates are the corresponding values of the standard normal distribution for the ai coefficients. These values are pre-calculated and available in statistical software packages.

        1. Calculate the numerator of the test statistic: This is the sum of the product of the ai coefficients and the corresponding $z_i$ normal deviates:

            $$
            \\sum (a_i \\times z_i)
            $$

        1. Calculate the denominator of the test statistic: This is the sum of the squared differences between the ordered sample values and the sample mean:

            $$
            \\sum (x_i - \\bar{x})^2
            $$

        1. Calculate the test statistic: This is the ratio of the squared numerator to the denominator:

            $$
            W = \\frac { (\\sum(a_i \\times z_i))^2 } { \\sum(x_i - \\bar{x})^2 }
            $$

        1. Compare the test statistic to a critical value: If the test statistic is less than the critical value, we cannot reject the null hypothesis of normality and can conclude that the residuals follow a normal distribution. If the test statistic is greater than the critical value, we reject the null hypothesis of normality and conclude that the residuals do not follow a normal distribution.

        In summary, the Shapiro-Wilk test is a statistical test that evaluates normality of time series residuals by standardizing and ordering the residuals, calculating ai coefficients and corresponding $z_i$ normal deviates, and computing the test statistic using a ratio of the squared sum of $ai \\times zi$ to the sum of squared differences between the sample values and sample mean. Finally, we compare the test statistic to a critical value to determine whether the residuals follow a normal distribution or not.

    Params:
        x (ArrayLike):
            Array of sample data

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

        ```pycon {.py .python linenums="1" title="Python"}
        >>> # Import packages
        >>> import numpy as np
        >>> from scipy.stats import shapiro

        >>> # generate sine wave data
        >>> x = np.linspace(0, 2 * np.pi, 100)
        >>> data = np.sin(x)

        >>> # run the Shapiro-Wilk test on the data
        >>> statistic, p_value = shapiro(data)

        >>> # print the results
        >>> print(f"Shapiro-Wilk test statistic: {statistic:.3f}")
        >>> print(f"Shapiro-Wilk test p-value: {p_value:.3f}")

        >>> # check if null hypothesis is rejected
        >>> alpha = 0.05
        >>> if p_value < alpha:
        ...     print("Reject null hypothesis that data is normally distributed")
        ... else:
        ...     print("Fail to reject null hypothesis that data is normally distributed")
        ...
        ```

        The null hypothesis of the Shapiro-Wilk test _is_ that the data is normally distributed. In this case, the p-value is `0.002`, which is **less** than the significance level of `0.05`, indicating that we _can_ reject the null hypothesis. Therefore, we can conclude that the sine wave data is **not** normally distributed.

        ---

        Example three, using the `FractionalGaussianNoise` random data generated from the `stochastic` package.

        ```pycon {.py .python linenums="1" title="Python"}
        >>> # Import packages
        >>> from stochastic.noise import FractionalGaussianNoise
        >>> from scipy.stats import shapiro

        >>> # Generate Fractional Gaussian Noise
        >>> fgn = FractionalGaussianNoise(t=1, hurst=0.5, length=100, method="daviesharte")
        >>> data = fgn.sample()

        >>> # run the Shapiro-Wilk test on the data
        >>> statistic, p_value = shapiro(data)

        >>> # print the results
        >>> print(f"Shapiro-Wilk test statistic: {statistic:.3f}")
        Shapiro-Wilk test statistic: 0.979
        >>> print(f"Shapiro-Wilk test p-value: {p_value:.3f}")
        Shapiro-Wilk test p-value: 0.417

        >>> # check if null hypothesis is rejected
        >>> alpha = 0.05
        >>> if p_value < alpha:
        ...     print("Reject null hypothesis that data is normally distributed")
        ... else:
        ...     print("Fail to reject null hypothesis that data is normally distributed")
        ...
        Fail to reject null hypothesis that data is normally distributed
        ```

        The null hypothesis of the Shapiro-Wilk test _is_ that the data is normally distributed. In this case, the p-value is `0.417`, which is **greater** than the significance level of `0.05`, indicating that we _fail_ to reject the null hypothesis. Therefore, we can conclude that the random noise generated by the `FractionalGaussianNoise` class **is** likely normally distributed.

    ??? note "Notes"
        The algorithm used is described in (Algorithm as R94 Appl. Statist. (1995)) but censoring parameters as described are not implemented. For $N > 5000$ the $W$ test statistic is accurate but the $p-value$ may not be.

        The chance of rejecting the null hypothesis when it is true is close to $5%$ regardless of sample size.

    ??? success "Credit"
        - All credit goes to the [`scipy`](https://docs.scipy.org/) library.

    ??? question "References"
        - https://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm
        - Shapiro, S. S. & Wilk, M.B (1965). An analysis of variance test for normality (complete samples), Biometrika, Vol. 52, pp. 591-611.
        - Razali, N. M. & Wah, Y. B. (2011) Power comparisons of Shapiro-Wilk, Kolmogorov-Smirnov, Lilliefors and Anderson-Darling tests, Journal of Statistical Modeling and Analytics, Vol. 2, pp. 21-33.
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
        The D'Agostino and Pearson's test is a statistical test used to evaluate whether a dataset, including time series data, follows a normal distribution. In time series forecasting, the D'Agostino and Pearson's test can be used to assess whether the residuals of a model follow a normal distribution, which is an assumption of many statistical models.

    ???+ abstract "Details"
        The D'Agostino and Pearson's test uses a combination of skewness and kurtosis measures to assess whether the residuals follow a normal distribution. Skewness measures the degree of asymmetry in the distribution of the residuals, while kurtosis measures the degree of peakedness or flatness. If the residuals follow a normal distribution, their skewness and kurtosis should be close to zero.

        To apply the D'Agostino and Pearson's test to time series data, we first need to estimate the residuals of the forecasting model. We can then use a statistical software package to perform the test on the residuals. The test produces a test statistic that compares the observed skewness and kurtosis values to the expected values under the null hypothesis of normality. If the observed values deviate significantly from the expected values under normality, the test rejects the null hypothesis and concludes that the residuals do not follow a normal distribution.

        The D'Agostino and Pearson's test produces a p-value, which indicates the probability of observing the observed test statistic if the null hypothesis of normality is true. If the p-value is less than the significance level (usually 0.05), we can conclude that the residuals do not follow a normal distribution.

        If the D'Agostino and Pearson's test indicates that the residuals do not follow a normal distribution, we may need to consider using a different modeling approach or modifying the forecasting model. It is important to ensure that the residuals of a time series forecasting model follow a normal distribution to ensure that the model is valid and reliable for making predictions.

    Params:
        x (ArrayLike):
            The array containing the sample to be tested.
        axis (int):
            Axis along which to compute test. If `None`, compute over the whole array `a`.<br>
            Defaults to `0`.
        nan_policy (VALID_DP_NAN_POLICY_OPTIONS):
            Defines how to handle when input contains nan. The following options are available (default is 'propagate'):

            - 'propagate': returns nan
            - 'raise': throws an error
            - 'omit': performs the calculations ignoring nan values.

            Defaults to `"propagate"`.

    Returns:
        statistic (Union[float, np.ndarray]):
            Value $s^2 + k^2$, where $s$ is the z-score returned by [`skewtest`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewtest.html) and $k$ is the z-score returned by [`kurtosistest`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosistest.html).
        pvalue (Union[float, np.ndarray]):
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

        ```pycon {.py .python linenums="1" title="Python"}
        >>> # Import packages
        >>> from stochastic import FractionalGaussianNoise
        >>> from scipy.stats import normaltest

        >>> # generate random noise using FractionalGaussianNoise
        >>> fgn = FractionalGaussianNoise(H=0.7, length=100)
        >>> data = fgn.generate()

        >>> # run D'Agostino and Pearson's test on the data
        >>> statistic, p_value = normaltest(data)

        >>> # print the results
        >>> print(f"D'Agostino and Pearson's test statistic: {statistic:.3f}")
        D'Agostino and Pearson's test statistic: 0.388
        >>> print(f"D'Agostino and Pearson's test p-value: {p_value:.3f}")
        D'Agostino and Pearson's test p-value: 0.823

        >>> # check if null hypothesis is rejected
        >>> alpha = 0.05
        >>> if p_value < alpha:
        ...     print("Reject null hypothesis that data is normally distributed")
        ... else:
        ...     print("Fail to reject null hypothesis that data is normally distributed")
        ...
        Fail to reject null hypothesis that data is normally distributed
        ```

        The null hypothesis of D'Agostino and Pearson's test is that the data _is_ normally distributed. In this case, the p-value is `0.823`, which is *greater* than the significance level of `0.05`, indicating that we _fail_ to reject the null hypothesis. Therefore, we can conclude that the random noise generated by the FractionalGaussianNoise class from the stochastic library *is* likely normally distributed.

    ??? note "Notes"
        This function is a wrapper for the `scipy.stats.normaltest` function.

        The D'Agostino and Pearson's test is a statistical test used to determine if a dataset, including time series data, is normally distributed. The test is based on the sample skewness and sample kurtosis of the dataset. The mathematical equation for the D'Agostino and Pearson's test is:

        $$
        D^2 = \\left( \\frac{n+1}{6} \\right) \\times \\left( S^2 + K^2 \\right)
        $$

        where:

        - $D^2$ is the test statistic
        - $n$ is the sample size
        - $S$ is the sample skewness
        - $K$ is the sample kurtosis

        To calculate the D'Agostino and Pearson's test statistic for time series data, we need to perform the following steps:

        1. Estimate the residuals of the forecasting model: The residuals are the difference between the actual values and the predicted values of the time series model.

        1. Calculate the sample mean and standard deviation of the residuals: These are the mean and standard deviation of the residuals, respectively.

        1. Standardize the residuals: The standardized residuals are the residuals divided by their sample standard deviation.

        1. Calculate the sample skewness: The sample skewness is a measure of the asymmetry of the distribution of the residuals. It is calculated as:

            $$
            S = \\left( \\frac {n} {(n-1) \\times (n-2)} \\right) \\times \\left( \\frac {\\sum_{i=1}^{n}(x_i - \\bar{x})^3 } { s^3 } \\right)
            $$

            where:

            - $x_i$ are the standardized residuals,
            - $\\bar{x}$ is their mean,
            - $s$ is their standard deviation, and
            - $n$ is the sample size.

        1. Calculate the sample kurtosis: The sample kurtosis is a measure of the "peakedness" of the distribution of the residuals. It is calculated as:

            $$
            K = \\left( \\frac { n \\times (n+1) } { (n-1) \\times (n-2) \\times (n-3) } \\right) \\times \\left( \\frac { \\sum_{i=1}^{n} (x_i - \\bar{x})^4 } { s^4 } \\right) - \\left( \\frac { 3 \\times (n-1)^2 } { (n-2) \\times (n-3) } \\right)
            $$

            where:

            - $x_i$ are the standardized residuals,
            - $\\bar{x}$ is their mean,
            - $s$ is their standard deviation, and
            - $n$ is the sample size.

        1. Calculate the test statistic: The test statistic is calculated using the formula:

            $$
            D^2 = \\left( \\frac { n+1 } {6} \\right) \\times \\left( S^2 + K^2 \\right)
            $$

        1. Compare the test statistic to a critical value: If the test statistic is less than the critical value, we cannot reject the null hypothesis of normality and can conclude that the residuals follow a normal distribution. If the test statistic is greater than the critical value, we reject the null hypothesis of normality and conclude that the residuals do not follow a normal distribution.

        In summary, the D'Agostino and Pearson's test is a statistical test that evaluates normality of time series residuals by standardizing the residuals, calculating their sample skewness and sample kurtosis, and computing the test statistic using a formula that takes into account both skewness and kurtosis. Finally, we compare the test statistic to a critical value to determine whether the residuals follow a normal distribution or not.

    ??? success "Credit"
        - All credit goes to the [`scipy`](https://docs.scipy.org/) library.

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
        The Anderson-Darling test is a statistical test used to evaluate whether a dataset, including time series data, follows a normal distribution. In time series forecasting, the Anderson-Darling test can be used to assess whether the residuals of a model follow a normal distribution, which is an assumption of many statistical models.

        The Anderson-Darling test tests the null hypothesis that a sample is drawn from a population that follows a particular distribution. For the Anderson-Darling test, the critical values depend on which distribution is being tested against. This function works for normal, exponential, logistic, or Gumbel (Extreme Value Type I) distributions.

    ???+ abstract "Details"
        The Anderson-Darling test is based on the null hypothesis that the residuals of the forecasting model are normally distributed. The test calculates a test statistic that measures the distance between the observed distribution of the residuals and the expected distribution under the null hypothesis of normality. If the observed distribution of the residuals deviates significantly from the expected distribution under normality, the test rejects the null hypothesis and concludes that the residuals do not follow a normal distribution.

        To apply the Anderson-Darling test to time series data, we first need to estimate the residuals of the forecasting model. We can then use a statistical software package to perform the test on the residuals. The test produces a p-value, which indicates the probability of observing the observed distribution of the residuals if the null hypothesis of normality is true. If the p-value is less than the significance level (usually $0.05$), we can conclude that the residuals do not follow a normal distribution.

        The Anderson-Darling test is more sensitive to deviations from normality in the tails of the distribution than other tests, such as the Shapiro-Wilk test. This makes it a useful test when assessing whether the residuals of a time series forecasting model exhibit heavy tails or other non-normal features.

        If the Anderson-Darling test indicates that the residuals do not follow a normal distribution, we may need to consider using a different modeling approach or modifying the forecasting model. It is important to ensure that the residuals of a time series forecasting model follow a normal distribution to ensure that the model is valid and reliable for making predictions.

    Params:
        x (ArrayLike):
            Array of sample data.
        dist (VALID_AD_DIST_OPTIONS):
            The type of distribution to test against. The default is `'norm'`. The names `'extreme1'`, `'gumbel_l'` and `'gumbel'` are synonyms for the same distribution.<br>
            Defaults to `"norm"`.

    Returns:
        statistic (float):
            The Anderson-Darling test statistic.
        critical_values (list):
            The critical values for this distribution.
        significance_level (list):
            The significance levels for the corresponding critical values in percents. The function returns critical values for a differing set of significance levels depending on the distribution that is being tested against.
        fit_result (Any):
            An object containing the results of fitting the distribution to the data.
            Note that the [`FitResult`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.FitResult.html) class was added to SciPy in version `1.10.0`. In the same release, this `anderson` function from SciPy had the outputs extended to include the `fit_result` object. The SciPy version `1.10.0` requires Python version `>=3.8`.<br><br>Therefore, when this function is executed on Python `3.7`, it will default to the highest compatible SciPy version, which is `1.7.0`. Hence, to ensure that this algorithm can still be used in Python `3.7`, the type of this output object is changed to `Any`, and is only returned when the Python version is `>=3.8`.

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

        ```pycon {.py .python linenums="1" title="Python"}
        >>> # Import packages
        >>> import numpy as np
        >>> from scipy.stats import anderson

        >>> # Generate sine wave data
        >>> x = np.sin(np.linspace(0, 2 * np.pi, 100))

        >>> # Perform Anderson-Darling test
        >>> result = anderson(x)

        >>> print("Statistic: %.3f" % result.statistic)
        Statistic: 0.161
        >>> for i in range(len(result.critical_values)):
        ...     sl, cv = result.significance_level[i], result.critical_values[i]
        ...     if result.statistic < cv:
        ...         print("%.3f: %.3f, data looks normal (fail to reject H0)" % (sl, cv))
        ...     else:
        ...         print("%.3f: %.3f, data does not look normal (reject H0)" % (sl, cv))
        ...
        15.000: 0.561, data looks normal (fail to reject H0)
        10.000: 0.638, data looks normal (fail to reject H0)
        5.000: 0.765, data looks normal (fail to reject H0)
        2.500: 0.892, data looks normal (fail to reject H0)
        1.000: 1.061, data looks normal (fail to reject H0)
        ```

        In this case, the Anderson-Darling test statistic is 0.161. The critical values and significance levels are also printed. The null hypothesis is that the data is drawn from a normal distribution. Based on the output, we can see that the statistic value is less than all of the critical values for the chosen significance levels, meaning we fail to reject the null hypothesis. Therefore, we can conclude that the sine wave data generated from NumPy is normally distributed.

        ---

        Example three, using the `FractionalGaussianNoise` random data generated from the `stochastic` package.

        ```pycon {.py .python linenums="1" title="Python"}
        >>> from stochastic.noise import FractionalGaussianNoise
        >>> from scipy.stats import anderson

        >>> # generate fractional Gaussian noise
        >>> fgn = FractionalGaussianNoise(t=1000, hurst=0.5)

        >>> # calculate the anderson-darling test
        >>> result = anderson(fgn.fgn(), dist="norm")

        >>> # print the result
        >>> print("Statistic: %.3f" % result.statistic)
        >>> p = 0
        >>> for i in range(len(result.critical_values)):
        ...     sl, cv = result.significance_level[i], result.critical_values[i]
        ...     if result.statistic < result.critical_values[i]:
        ...         print("%.3f: %.3f, data looks normal (fail to reject H0)" % (sl, cv))
        ...     else:
        ...         print("%.3f: %.3f, data does not look normal (reject H0)" % (sl, cv))
        ...
        ```

        In this example, the test statistic is 1.171, and the critical values at different significance levels are 0.570, 0.648, 0.777, 0.906, and 1.076. Since the test statistic is greater than all of the critical values, we can reject the null hypothesis and conclude that the data is not normally distributed.

    ??? note "Notes"
        Critical values provided are for the following significance levels:

        - normal/exponential
            - 15%, 10%, 5%, 2.5%, 1%
        - logistic
            - 25%, 10%, 5%, 2.5%, 1%, 0.5%
        - Gumbel
            - 25%, 10%, 5%, 2.5%, 1%

        If the returned statistic is larger than these critical values then for the corresponding significance level, the null hypothesis that the data come from the chosen distribution can be rejected. The returned statistic is referred to as 'A2' in the references.

        The Anderson-Darling test is a statistical test used to determine whether a dataset, including time series data, is normally distributed. The test is based on the deviations of the sample distribution from the theoretical normal distribution. The mathematical equation for the Anderson-Darling test is:

        $$
        A^2 = -n - \\sum_{i=1}^{n} \\left( \\left( \\frac{2 \\times i - 1}{n} \\right) \\times (log(F(x_i)) + log(1-F(n-i+1))) \\right)
        $$

        where:

        - $A^2$ is the test statistic
        - $n$ is the sample size
        - $F(x_i)$ is the empirical distribution function of the sample at $x_i$
        - $log$ is a natural logarithm

        To calculate the Anderson-Darling test statistic for time series data, we need to perform the following steps:

        1. Estimate the residuals of the forecasting model: The residuals are the difference between the actual values and the predicted values of the time series model.

        1. Calculate the sample mean and standard deviation of the residuals: These are the mean and standard deviation of the residuals, respectively.

        1. Standardize the residuals: The standardized residuals are the residuals divided by their sample standard deviation.

        1. Sort the standardized residuals in ascending order.

        1. Calculate the empirical distribution function (EDF) of the residuals: The EDF is the proportion of the standardized residuals that are less than or equal to a given value. It is calculated as:

            $$
            F(x_i) = \\left( \\frac{1}{n} \\right) \\times \\sum_{j=1}^{n} \\left( I(x_i <= x_j) \\right)
            $$

            where:

            - $x_i$ are the sorted standardized residuals,
            - $x_j$ are the sample values, and
            - $I()$ is the indicator function.

        1. Calculate the test statistic: The test statistic is calculated using the formula:

            $$
            A^2 = -n - \\sum_{i=1}^{n} \\left( \\left( \\frac{2 \\times i - 1}{n} \\right) \\times (log(F(x_i)) + log(1-F(n-i+1))) \\right)
            $$

            where:

            - $x_i$ are the sorted standardized residuals.

        1. Compare the test statistic to a critical value: If the test statistic is less than the critical value, we cannot reject the null hypothesis of normality and can conclude that the residuals follow a normal distribution. If the test statistic is greater than the critical value, we reject the null hypothesis of normality and conclude that the residuals do not follow a normal distribution.

        In summary, the Anderson-Darling test is a statistical test that evaluates normality of time series residuals by comparing the empirical distribution function of the sample data to the cumulative distribution function of the normal distribution. The test statistic is calculated by summing the product of weights and logarithms of the empirical distribution function and the complement of the normal distribution CDF. Finally, we compare the test statistic to a critical value to determine whether the residuals follow a normal distribution or not.

    ??? success "Credit"
        - All credit goes to the [`scipy`](https://docs.scipy.org/) library.

    ??? question "References"
        - https://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm
        - Stephens, M. A. (1974). EDF Statistics for Goodness of Fit and Some Comparisons, Journal of the American Statistical Association, Vol. 69, pp. 730-737.
        - Stephens, M. A. (1976). Asymptotic Results for Goodness-of-Fit Statistics with Unknown Parameters, Annals of Statistics, Vol. 4, pp. 357-369.
        - Stephens, M. A. (1977). Goodness of Fit for the Extreme Value Distribution, Biometrika, Vol. 64, pp. 583-588.
        - Stephens, M. A. (1977). Goodness of Fit with Special Reference to Tests for Exponentiality , Technical Report No. 262, Department of Statistics, Stanford University, Stanford, CA.
        - Stephens, M. A. (1979). Tests of Fit for the Logistic Distribution Based on the Empirical Distribution Function, Biometrika, Vol. 66, pp. 591-595.

    ??? tip "See Also"
        - [`jb()`][ts_stat_tests.algorithms.normality.jb]
        - [`ob()`][ts_stat_tests.algorithms.normality.ob]
        - [`sw()`][ts_stat_tests.algorithms.normality.sw]
        - [`dp()`][ts_stat_tests.algorithms.normality.dp]
    """
    return _ad(x=x, dist=dist)
