# ============================================================================ #
#                                                                              #
#     Title: Heteroscedasticity Tests                                          #
#     Purpose: Implementation of heteroscedasticity test wrappers.             #
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
    This module provides wrapper functions for various heteroscedasticity tests including:
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
from typing import Any, Callable, Union

# ## Python Third Party Imports ----
from numpy.typing import ArrayLike
from statsmodels.regression.linear_model import (
    RegressionResults,
    RegressionResultsWrapper,
)
from typeguard import typechecked

# ## Local First Party Imports ----
from ts_stat_tests.heteroscedasticity.algorithms import arch, bpl, gq, wlm
from ts_stat_tests.utils.errors import generate_error_message


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["heteroscedasticity", "is_heteroscedastic"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Tests                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@typechecked
def heteroscedasticity(
    res: Union[RegressionResults, RegressionResultsWrapper],
    algorithm: str = "bp",
    **kwargs: Union[float, int, str, bool, ArrayLike, None],
) -> tuple[Any, ...]:
    """
    !!! note "Summary"
        Perform a heteroscedasticity test on a fitted regression model.

    ???+ abstract "Details"
        This function is a convenience wrapper around four underlying algorithms:<br>
        - [`arch()`][ts_stat_tests.heteroscedasticity.algorithms.arch]<br>
        - [`bp()`][ts_stat_tests.heteroscedasticity.algorithms.bpl]<br>
        - [`gq()`][ts_stat_tests.heteroscedasticity.algorithms.gq]<br>
        - [`white()`][ts_stat_tests.heteroscedasticity.algorithms.wlm]

    Params:
        res (Union[RegressionResults, RegressionResultsWrapper]):
            The fitted regression model to be checked.
        algorithm (str):
            Which heteroscedasticity algorithm to use.<br>
            - `arch()`: `["arch", "engle"]`<br>
            - `bp()`: `["bp", "breusch-pagan", "breusch-pagan-lagrange-multiplier"]`<br>
            - `gq()`: `["gq", "goldfeld-quandt"]`<br>
            - `white()`: `["white"]`<br>
            Default: `"bp"`
        kwargs (Union[float, int, str, bool, ArrayLike, None]):
            Additional keyword arguments passed to the underlying test function.

    Raises:
        (ValueError):
            When the given value for `algorithm` is not valid.

    Returns:
        (Union[tuple[float, float, float, float], tuple[float, float, str], ResultsStore]):
            The results of the heteroscedasticity test. The return type depends on the chosen algorithm and `kwargs`.

    !!! success "Credit"
        Calculations are performed by `statsmodels`.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> import statsmodels.api as sm
        >>> from ts_stat_tests.heteroscedasticity.tests import heteroscedasticity
        >>> from ts_stat_tests.utils.data import data_line, data_random
        >>> X = sm.add_constant(data_line)
        >>> y = 2 * data_line + data_random
        >>> res = sm.OLS(y, X).fit()

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Breusch-Pagan test"}
        >>> result = heteroscedasticity(res, algorithm="bp")
        >>> print(f"p-value: {result[1]:.4f}")
        p-value: 0.2461

        ```

        ```pycon {.py .python linenums="1" title="Example 2: ARCH test"}
        >>> lm, lmp, f, fp = heteroscedasticity(res, algorithm="arch")
        >>> print(f"ARCH p-value: {lmp:.4f}")
        ARCH p-value: 0.9124

        ```
    """
    options: dict[str, tuple[str, ...]] = {
        "arch": ("arch", "engle"),
        "bp": ("bp", "breusch-pagan", "breusch-pagan-lagrange-multiplier"),
        "gq": ("gq", "goldfeld-quandt"),
        "white": ("white",),
    }

    # Internal helper to handle kwargs casting for ty
    def _call(
        func: Callable[..., Any],
        **args: Any,
    ) -> tuple[Any, ...]:
        """
        !!! note "Summary"
            Internal helper to handle keyword arguments types.

        Params:
            func (Callable[..., Any]):
                The function to call.
            args (Any):
                The keyword arguments to pass.

        Returns:
            (tuple[Any, ...]):
                The function output.

        ???+ example "Examples"
            This is an internal function and is not intended to be called directly.
        """
        return func(**args)

    if algorithm in options["arch"]:
        return _call(arch, resid=res.resid, **kwargs)

    if algorithm in options["bp"]:
        return _call(bpl, resid=res.resid, exog_het=res.model.exog, **kwargs)

    if algorithm in options["gq"]:
        return _call(gq, y=res.model.endog, x=res.model.exog, **kwargs)

    if algorithm in options["white"]:
        return _call(wlm, resid=res.resid, exog_het=res.model.exog, **kwargs)

    raise ValueError(
        generate_error_message(
            parameter_name="algorithm",
            value_parsed=algorithm,
            options=options,
        )
    )


@typechecked
def is_heteroscedastic(
    res: Union[RegressionResults, RegressionResultsWrapper],
    algorithm: str = "bp",
    alpha: float = 0.05,
    **kwargs: Union[float, int, str, bool, ArrayLike, None],
) -> dict[str, Union[str, float, bool, None]]:
    """
    !!! note "Summary"
        Test whether a given model's residuals exhibit `heteroscedasticity` or not.

    ???+ abstract "Details"
        This function checks the results of a heteroscedasticity test against a significance level `alpha`. The null hypothesis ($H_0$) for all supported tests is homoscedasticity (constant variance). If the p-value is less than `alpha`, the null hypothesis is rejected in favor of heteroscedasticity.

    Params:
        res (Union[RegressionResults, RegressionResultsWrapper]):
            The fitted regression model to be checked.
        algorithm (str):
            Which heteroscedasticity algorithm to use. See [`heteroscedasticity()`][ts_stat_tests.heteroscedasticity.tests.heteroscedasticity] for options.
            Default: `"bp"`
        alpha (float):
            The significance level for the test.
            Default: `0.05`
        kwargs (Union[float, int, str, bool, ArrayLike, None]):
            Additional keyword arguments passed to the underlying test function.

    Returns:
        (dict[str, Union[str, float, bool, None]]):
            A dictionary containing:
            - `"result"` (bool): Indicator if the residuals are heteroscedastic (i.e., p-value < alpha).
            - `"statistic"` (float): The test statistic.
            - `"pvalue"` (float): The p-value of the test.
            - `"alpha"` (float): The significance level used.
            - `"algorithm"` (str): The algorithm used.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> import statsmodels.api as sm
        >>> from ts_stat_tests.heteroscedasticity.tests import is_heteroscedastic
        >>> from ts_stat_tests.utils.data import data_line, data_random
        >>> X = sm.add_constant(data_line)
        >>> y = 2 * data_line + data_random
        >>> res = sm.OLS(y, X).fit()

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Check heteroscedasticity with Breusch-Pagan"}
        >>> res_check = is_heteroscedastic(res, algorithm="bp")
        >>> print(res_check["result"])
        False

        ```
    """
    raw_res = heteroscedasticity(res=res, algorithm=algorithm, **kwargs)

    # All heteroscedasticity algorithms return a tuple
    # (lm, lmpval, fval, fpval) or (fval, pval, ...)
    stat = float(raw_res[0])
    pvalue = float(raw_res[1])

    return {
        "algorithm": algorithm,
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "alpha": alpha,
        "result": bool(pvalue < alpha),
    }
