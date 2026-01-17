# ============================================================================ #
#                                                                              #
#     Title: Linearity Algorithms                                              #
#     Purpose: Implementation of linearity test algorithms using statsmodels.  #
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
    This module provides implementations of various linearity test algorithms using the `statsmodels` library.
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
from typing import Callable, Literal, Optional, Union

# ## Python Third Party Imports ----
import numpy as np
from numpy.typing import ArrayLike, NDArray
from statsmodels.regression.linear_model import (
    RegressionResults,
    RegressionResultsWrapper,
)
from statsmodels.stats.api import (
    linear_harvey_collier,
    linear_lm,
    linear_rainbow,
    linear_reset,
)
from statsmodels.stats.contrast import ContrastResults
from typeguard import typechecked


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["hc", "lm", "rb", "rr"]


## --------------------------------------------------------------------------- #
##  Constants                                                               ####
## --------------------------------------------------------------------------- #


VALID_RR_TEST_TYPE_OPTIONS = Literal["fitted", "exog", "princomp"]
VALID_RR_COV_TYPE_OPTIONS = Literal["nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Algorithms                                                             ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@typechecked
def hc(
    res: Union[RegressionResults, RegressionResultsWrapper],
    order_by: Optional[ArrayLike] = None,
    skip: Optional[int] = None,
) -> tuple[float, float]:
    r"""
    !!! note "Summary"
        The Harvey-Collier test is a statistical test used to determine whether a dataset follows a linear relationship. In time series forecasting, the test can be used to evaluate whether the residuals of a model follow a linear distribution.

    ???+ abstract "Details"
        The Harvey-Collier test is based on a recursive residuals analysis. The test statistic follows a t-distribution under the null hypothesis of linearity.

    Params:
        res (Union[RegressionResults, RegressionResultsWrapper]):
            The results of a linear regression model from `statsmodels`.
        order_by (Optional[ArrayLike]):
            Variable(s) to order by. If `None`, the original order is used.
        skip (Optional[int]):
            The number of observations to skip at the beginning of the series.

    Returns:
        (tuple[float, float]):
            - `statistic` (float): The t-statistic of the test.
            - `pvalue` (float): The p-value associated with the t-statistic.

    ???+ example "Examples"
        ```python
        import statsmodels.api as sm
        from ts_stat_tests.algorithms.linearity import hc

        X = sm.add_constant(np.arange(100))
        y = 3 + 2 * X[:, 1] + np.random.normal(size=100)
        res = sm.OLS(y, X).fit()
        hc(res)
        ```

    ??? question "References"
        - Harvey, A.C. and Collier, P. (1977). "Testing for Functional Form in Regression with Application to an Agricultural Production Function." Journal of Econometrics, 6(1), 103-119.
    """
    res_hc = linear_harvey_collier(res=res, order_by=order_by, skip=skip)
    return float(getattr(res_hc, "statistic", np.nan)), float(getattr(res_hc, "pvalue", np.nan))


@typechecked
def lm(
    resid: NDArray[np.float64], exog: NDArray[np.float64], func: Optional[Callable] = None
) -> tuple[float, float, float, float]:
    r"""
    !!! note "Summary"
        Lagrange Multiplier test for functional form / linearity.

    ???+ abstract "Details"
        This test checks whether the linear specification is appropriate for the data. It is a general test for functional form misspecification.

    Params:
        resid (NDArray[np.float64]):
            The residuals from a linear regression.
        exog (NDArray[np.float64]):
            The exogenous variables (predictors) used in the regression.
        func (Optional[Callable]):
            A function that takes `exog` and returns a transformed version of it to test against.
            Default: `None`

    Returns:
        (tuple[float, float, float, float]):
            - `lm` (float): Lagrange multiplier statistic.
            - `lmpval` (float): p-value for LM statistic.
            - `fval` (float): F-statistic.
            - `fpval` (float): p-value for F-statistic.

    ???+ example "Examples"
        ```python
        import numpy as np
        from ts_stat_tests.algorithms.linearity import lm

        resid = np.random.normal(size=100)
        exog = np.random.normal(size=(100, 2))
        lm(resid, exog)
        ```
    """
    res_lm = linear_lm(resid=resid, exog=exog, func=func)
    return (
        float(res_lm[0]),
        float(res_lm[1]),
        float(getattr(res_lm[2], "fvalue", np.nan)),
        float(getattr(res_lm[2], "pvalue", np.nan)),
    )


@typechecked
def rb(
    res: Union[RegressionResults, RegressionResultsWrapper],
    frac: float = 0.5,
    order_by: Optional[Union[ArrayLike, str, list[str]]] = None,
    use_distance: bool = False,
    center: Optional[Union[float, int]] = None,
) -> tuple[float, float]:
    r"""
    !!! note "Summary"
        The Rainbow test for linearity.

    ???+ abstract "Details"
        The Rainbow test is a test for linearity that is based on the idea that if a relationship is non-linear, it is more likely to be linear in a subset of the data than in the entire dataset.

    Params:
        res (Union[RegressionResults, RegressionResultsWrapper]):
            The results of a linear regression model from `statsmodels`.
        frac (float):
            The fraction of the data to use for the subset.
            Default: `0.5`
        order_by (Optional[Union[ArrayLike, str, list[str]]]):
            Variable(s) to order by. If `None`, the original order is used.
        use_distance (bool):
            Whether to use distance from the center for ordering.
            Default: `False`
        center (Optional[Union[float, int]]):
            The center to use for distance calculation.
            Default: `None`

    Returns:
        (tuple[float, float]):
            - `fstat` (float): The F-statistic of the test.
            - `pvalue` (float): The p-value associated with the F-statistic.

    ???+ example "Examples"
        ```python
        import statsmodels.api as sm
        from ts_stat_tests.algorithms.linearity import rb

        X = sm.add_constant(np.arange(100))
        y = 3 + 2 * X[:, 1] + 2 * X[:, 1] ** 2 + np.random.normal(size=100)
        res = sm.OLS(y, X).fit()
        rb(res)
        ```

    ??? question "References"
        - Utts, J.M. (1982). "The Rainbow Test for Linearity." Biometrika, 69(2), 319-326.
    """
    res_rb = linear_rainbow(res=res, frac=frac, order_by=order_by, use_distance=use_distance, center=center)
    return float(res_rb[0]), float(res_rb[1])


@typechecked
def rr(
    res: Union[RegressionResults, RegressionResultsWrapper],
    power: Union[int, list[int]] = 3,
    test_type: VALID_RR_TEST_TYPE_OPTIONS = "fitted",
    use_f: bool = False,
    cov_type: VALID_RR_COV_TYPE_OPTIONS = "nonrobust",
    *,
    cov_kwargs: Optional[dict] = None,
) -> ContrastResults:
    r"""
    !!! note "Summary"
        Ramsey's RESET (Regression Specification Error Test) for linearity.

    ???+ abstract "Details"
        RESET test for functional form misspecification. The test is based on the idea that if the model is correctly specified, then powers of the fitted values (or other variables) should not have any explanatory power when added to the model.

    Params:
        res (Union[RegressionResults, RegressionResultsWrapper]):
            The results of a linear regression model from `statsmodels`.
        power (Union[int, list[int]]):
            The powers of the fitted values or exogenous variables to include in the auxiliary regression.
            Default: `3`
        test_type (VALID_RR_TEST_TYPE_OPTIONS):
            The type of test to perform. Options are `"fitted"`, `"exog"`, or `"princomp"`.
            Default: `"fitted"`
        use_f (bool):
            Whether to use an F-test or a Chi-squared test.
            Default: `False`
        cov_type (VALID_RR_COV_TYPE_OPTIONS):
            The type of covariance matrix to use in the test.
            Default: `"nonrobust"`
        cov_kwargs (Optional[dict]):
            Optional keyword arguments for the covariance matrix calculation.
            Default: `None`

    Returns:
        (ContrastResults):
            The results of the RESET test.

    ???+ example "Examples"
        ```python
        import statsmodels.api as sm
        from ts_stat_tests.algorithms.linearity import rr

        X = sm.add_constant(np.arange(100))
        y = 3 + 2 * X[:, 1] + np.random.normal(size=100)
        res = sm.OLS(y, X).fit()
        rr(res)
        ```

    ??? question "References"
        - Ramsey, J.B. (1969). "Tests for Specification Errors in Classical Linear Least-squares Regression Analysis." Journal of the Royal Statistical Society, Series B, 31(2), 350-371.
    """
    return linear_reset(
        res=res,
        power=power,  # type: ignore[arg-type]
        test_type=test_type,
        use_f=use_f,
        cov_type=cov_type,
        cov_kwargs=cov_kwargs,
    )
