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
from typing import (
    Literal,
    Optional,
    Union,
    cast,
    overload,
)

# ## Python Third Party Imports ----
from numpy.typing import ArrayLike
from statsmodels.stats.diagnostic import (
    ResultsStore,
    het_arch,
    het_breuschpagan,
    het_goldfeldquandt,
    het_white,
)
from typeguard import typechecked


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["arch", "bpl", "gq", "wlm"]


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
) -> tuple[float, float, float, float]: ...
@overload
def arch(
    resid: ArrayLike, nlags: Optional[int] = None, ddof: int = 0, *, store: Literal[True]
) -> tuple[float, float, float, float, ResultsStore]: ...
@typechecked
def arch(resid: ArrayLike, nlags: Optional[int] = None, ddof: int = 0, *, store: bool = False) -> Union[
    tuple[float, float, float, float],
    tuple[float, float, float, float, ResultsStore],
]:
    r"""
    !!! note "Summary"
        Engle's Test for Autoregressive Conditional Heteroscedasticity (ARCH).

    ???+ abstract "Details"
        This test is used to determine whether the residuals of a time-series model exhibit ARCH effects. ARCH effects are characterized by clusters of volatility, where periods of high volatility are followed by periods of high volatility, and vice versa. The test is essentially a Lagrange Multiplier (LM) test for autocorrelation in the squared residuals.

    Params:
        resid (ArrayLike):
            The residuals from a linear regression model.
        nlags (Optional[int]):
            The number of lags to include in the test regression. If `None`, the number of lags is determined based on the number of observations.
            Default: `None`
        ddof (int):
            Degrees of freedom to adjust for in the calculation of the F-statistic.
            Default: `0`
        store (bool):
            Whether to return a `ResultsStore` object containing additional test results.
            Default: `False`

    Returns:
        (Union[tuple[float, float, float, float], tuple[float, float, float, float, ResultsStore]]):
            A tuple containing:
            - `lmstat` (float): The Lagrange Multiplier statistic.
            - `lmpval` (float): The p-value for the LM statistic.
            - `fstat` (float): The F-statistic.
            - `fpval` (float): The p-value for the F-statistic.
            - `resstore` (ResultsStore, optional): Returned only if `store` is `True`.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> from ts_stat_tests.algorithms.heteroscedasticity import arch
        >>> x = np.linspace(0, 10, 100)
        >>> X = sm.add_constant(x)
        >>> y = 2 * x + np.random.normal(size=100)
        >>> res = sm.OLS(y, X).fit()
        >>> resid = res.resid

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Basic ARCH test"}
        >>> lm, lmp, f, fp = arch(resid)
        >>> print(f"LM p-value: {lmp:.4f}")  # doctest: +SKIP
        LM p-value: 0.4567

        ```

    ??? equation "Calculation"
        The test is performed by regressing the squared residuals $e_t^2$ on a constant and $q$ lags of the squared residuals:

        $$
        e_t^2 = \gamma_0 + \gamma_1 e_{t-1}^2 + \gamma_2 e_{t-2}^2 + \dots + \gamma_q e_{t-q}^2 + \nu_t
        $$

        The null hypothesis of no ARCH effects is:

        $$
        H_0: \gamma_1 = \gamma_2 = \dots = \gamma_q = 0
        $$

        The LM statistic is calculated as $T \times R^2$ from this regression, where $T$ is the number of observations and $R^2$ is the coefficient of determination.

    ??? success "Credit"
        Calculations are performed by `statsmodels`.

    ??? question "References"
        - Engle, R. F. (1982). Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation. Econometrica, 50(4), 987-1007.
    """
    if store:
        res_5 = cast(
            tuple[float, float, float, float, ResultsStore],
            het_arch(resid=resid, nlags=nlags, store=True, ddof=ddof),
        )
        return (
            float(res_5[0]),
            float(res_5[1]),
            float(res_5[2]),
            float(res_5[3]),
            res_5[4],
        )

    res_4 = cast(
        tuple[float, float, float, float],
        het_arch(resid=resid, nlags=nlags, store=False, ddof=ddof),
    )
    return (float(res_4[0]), float(res_4[1]), float(res_4[2]), float(res_4[3]))


@typechecked
def bpl(resid: ArrayLike, exog_het: ArrayLike, robust: bool = True) -> tuple[float, float, float, float]:
    r"""
    !!! note "Summary"
        Breusch-Pagan Lagrange Multiplier Test for Heteroscedasticity.

    ???+ abstract "Details"
        This test checks whether the variance of the errors in a regression model depends on the values of the independent variables. If it does, the errors are heteroscedastic. The null hypothesis assumes homoscedasticity (constant variance).

    Params:
        resid (ArrayLike):
            The residuals from a linear regression model.
        exog_het (ArrayLike):
            The explanatory variables for the variance (heteroscedasticity). Usually, these are the same as the original regression's exogenous variables.
        robust (bool):
            Whether to use a robust version of the test that does not assume the errors are normally distributed (Koenker's version).
            Default: `True`

    Returns:
        (tuple[float, float, float, float]):
            A tuple containing:
            - `lmstat` (float): The Lagrange Multiplier statistic.
            - `lmpval` (float): The p-value for the LM statistic.
            - `fstat` (float): The F-statistic.
            - `fpval` (float): The p-value for the F-statistic.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> from ts_stat_tests.algorithms.heteroscedasticity import bpl
        >>> x = np.linspace(0, 10, 100)
        >>> X = sm.add_constant(x)
        >>> y = 2 * x + np.random.normal(size=100)
        >>> res = sm.OLS(y, X).fit()
        >>> resid, exog = res.resid, X

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Basic Breusch-Pagan test"}
        >>> lm, lmp, f, fp = bpl(resid, exog)
        >>> print(f"LM p-value: {lmp:.4f}")  # doctest: +SKIP
        LM p-value: 0.1234

        ```

    ??? equation "Calculation"
        The test first fits a regression of squared residuals (or standardized version) on the specified exogenous variables:

        $$
        e_t^2 = \delta_0 + \delta_1 z_{t1} + \dots + \delta_k z_{tk} + u_t
        $$

        The null hypothesis is:

        $$
        H_0: \delta_1 = \dots = \delta_k = 0
        $$

        Koenker's robust version uses the scores of the likelihood function and does not require the normality assumption.

    ??? success "Credit"
        Calculations are performed by `statsmodels`.

    ??? question "References"
        - Breusch, T. S., & Pagan, A. R. (1979). A Simple Test for Heteroscedasticity and Random Coefficient Variation. Econometrica, 47(5), 1287-1294.
        - Koenker, R. (1981). A Note on Studentizing a Test for Heteroscedasticity. Journal of Econometrics, 17(1), 107-112.
    """
    res = het_breuschpagan(resid=resid, exog_het=exog_het, robust=robust)
    return (float(res[0]), float(res[1]), float(res[2]), float(res[3]))


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
) -> tuple[float, float, str]: ...
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
) -> tuple[float, float, str, ResultsStore]: ...
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
    tuple[float, float, str],
    tuple[float, float, str, ResultsStore],
]:
    r"""
    !!! note "Summary"
        Goldfeld-Quandt Test for Heteroscedasticity.

    ???+ abstract "Details"
        The Goldfeld-Quandt test checks for heteroscedasticity by dividing the dataset into two subsets (usually at the beginning and end of the sample) and comparing the variance of the residuals in each subset using an F-test.

    Params:
        y (ArrayLike):
            The dependent variable (endogenous).
        x (ArrayLike):
            The independent variables (exogenous).
        idx (Optional[int]):
            The column index of the variable to sort by. If `None`, the data is assumed to be ordered.
            Default: `None`
        split (Optional[Union[int, float]]):
            The index at which to split the sample. If a float between 0 and 1, it represents the fraction of observations.
            Default: `None`
        drop (Optional[Union[int, float]]):
            The number of observations to drop in the middle. If a float between 0 and 1, it represents the fraction of observations.
            Default: `None`
        alternative (VALID_GQ_ALTERNATIVES_OPTIONS):
            The alternative hypothesis. Options are `"increasing"`, `"decreasing"`, or `"two-sided"`.
            Default: `"increasing"`
        store (bool):
            Whether to return a `ResultsStore` object.
            Default: `False`

    Returns:
        (Union[tuple[float, float, str], tuple[float, float, str, ResultsStore]]):
            A tuple containing:
            - `fstat` (float): The F-statistic.
            - `fpval` (float): The p-value for the F-statistic.
            - `alternative` (str): The alternative hypothesis used.
            - `resstore` (ResultsStore, optional): Returned only if `store` is `True`.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> from ts_stat_tests.algorithms.heteroscedasticity import gq
        >>> x = np.linspace(0, 10, 100)
        >>> X = sm.add_constant(x)
        >>> y = 2 * x + np.random.normal(size=100)

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Basic Goldfeld-Quandt test"}
        >>> f, p, alt = gq(y, X)
        >>> print(f"F p-value: {p:.4f}")  # doctest: +SKIP
        F p-value: 0.5678

        ```

    ??? equation "Calculation"
        The dataset is split into two samples after sorting by an independent variable (or using the natural order). Separate regressions are run on each sample:

        $$
        RSS_1 = \sum e_{1,t}^2, \quad RSS_2 = \sum e_{2,t}^2
        $$

        The test statistic is the ratio of variances:

        $$
        F = \frac{RSS_2 / df_2}{RSS_1 / df_1}
        $$

        where $RSS_i$ are the residual sum of squares and $df_i$ are the degrees of freedom.

    ??? success "Credit"
        Calculations are performed by `statsmodels`.

    ??? question "References"
        - Goldfeld, S. M., & Quandt, R. E. (1965). Some Tests for Homoscedasticity. Journal of the American Statistical Association, 60(310), 539-547.
    """
    if store:
        res_4 = cast(
            tuple[float, float, str, ResultsStore],
            het_goldfeldquandt(
                y=y,
                x=x,
                idx=idx,
                split=split,
                drop=drop,
                alternative=alternative,
                store=True,
            ),
        )
        return (float(res_4[0]), float(res_4[1]), str(res_4[2]), res_4[3])

    res_3 = cast(
        tuple[float, float, str],
        het_goldfeldquandt(
            y=y,
            x=x,
            idx=idx,
            split=split,
            drop=drop,
            alternative=alternative,
            store=False,
        ),
    )
    return (float(res_3[0]), float(res_3[1]), str(res_3[2]))


@typechecked
def wlm(resid: ArrayLike, exog_het: ArrayLike) -> tuple[float, float, float, float]:
    r"""
    !!! note "Summary"
        White's Test for Heteroscedasticity.

    ???+ abstract "Details"
        White's test is a general test for heteroscedasticity that does not require a specific functional form for the variance of the error terms. It is essentially a test of whether the squared residuals can be explained by the levels, squares, and cross-products of the independent variables.

    Params:
        resid (ArrayLike):
            The residuals from a linear regression model.
        exog_het (ArrayLike):
            The explanatory variables for the variance. Usually, these are the original exogenous variables; the test internally handles adding their squares and cross-products.

    Returns:
        (tuple[float, float, float, float]):
            A tuple containing:
            - `lmstat` (float): The Lagrange Multiplier statistic.
            - `lmpval` (float): The p-value for the LM statistic.
            - `fstat` (float): The F-statistic.
            - `fpval` (float): The p-value for the F-statistic.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> from ts_stat_tests.algorithms.heteroscedasticity import wlm
        >>> x = np.linspace(0, 10, 100)
        >>> X = sm.add_constant(x)
        >>> y = 2 * x + np.random.normal(size=100)
        >>> res = sm.OLS(y, X).fit()
        >>> resid, exog = res.resid, X

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Basic White's test"}
        >>> lm, lmp, f, fp = wlm(resid, exog)
        >>> print(f"White p-value: {lmp:.4f}")  # doctest: +SKIP
        White p-value: 0.2345

        ```

    ??? equation "Calculation"
        Squared residuals are regressed on all distinct variables in the cross-product of the original exogenous variables (including constant, linear terms, squares, and interactions):

        $$
        e_t^2 = \delta_0 + \sum \delta_i z_{it} + \sum \delta_{ij} z_{it} z_{jt} + u_t
        $$

        The LM statistic is $T \times R^2$ from this auxiliary regression, where $T$ is the number of observations.

    ??? success "Credit"
        Calculations are performed by `statsmodels`.

    ??? question "References"
        - White, H. (1980). A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity. Econometrica, 48(4), 817-838.
    """
    res = het_white(resid=resid, exog=exog_het)
    return (float(res[0]), float(res[1]), float(res[2]), float(res[3]))
