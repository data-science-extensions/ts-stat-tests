# ============================================================================ #
#                                                                              #
#     Title: Linearity Tests                                                   #
#     Purpose: Convenience functions for linearity algorithms.                 #
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
    This module contains convenience functions and tests for linearity measures, allowing for easy access to different linearity algorithms.
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
from typing import Callable, Union

# ## Python Third Party Imports ----
import numpy as np
from numpy.typing import ArrayLike
from statsmodels.regression.linear_model import (
    RegressionResults,
    RegressionResultsWrapper,
)
from statsmodels.stats.contrast import ContrastResults
from typeguard import typechecked

# ## Local First Party Imports ----
from ts_stat_tests.algorithms.linearity import (
    hc as _hc,
    lm as _lm,
    rb as _rb,
    rr as _rr,
)
from ts_stat_tests.utils.errors import generate_error_message


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["linearity", "is_linear"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Tests                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@typechecked
def linearity(
    res: Union[RegressionResults, RegressionResultsWrapper],
    algorithm: str = "rr",
    **kwargs: Union[float, int, str, bool, ArrayLike, None],
) -> Union[tuple[float, ...], ContrastResults]:
    """
    !!! note "Summary"
        Perform a linearity test on a fitted regression model.

    ???+ abstract "Details"
        This function is a convenience wrapper around four underlying algorithms:<br>
        - [`hc()`][ts_stat_tests.algorithms.linearity.hc]<br>
        - [`lm()`][ts_stat_tests.algorithms.linearity.lm]<br>
        - [`rb()`][ts_stat_tests.algorithms.linearity.rb]<br>
        - [`rr()`][ts_stat_tests.algorithms.linearity.rr]

    Params:
        res (Union[RegressionResults, RegressionResultsWrapper]):
            The fitted regression model to be checked.
        algorithm (str):
            Which linearity algorithm to use.<br>
            - `hc()`: `["hc", "harvey", "harvey-collier"]`<br>
            - `lm()`: `["lm", "lagrange", "lagrange-multiplier"]`<br>
            - `rb()`: `["rb", "rainbow"]`<br>
            - `rr()`: `["rr", "reset", "ramsey-reset"]`<br>
            Default: `"rr"`
        kwargs (Union[float, int, str, bool, ArrayLike, None]):
            Additional keyword arguments passed to the underlying test function.

    Raises:
        (ValueError):
            When the given value for `algorithm` is not valid.

    Returns:
        (Union[tuple[float, float], tuple[float, float, float, float], ContrastResults]):
            - For `"hc"`: `(statistic, pvalue)`
            - For `"lm"`: `(lm_stat, lm_pval, f_stat, f_pval)`
            - For `"rb"`: `(statistic, pvalue)`
            - For `"rr"`: `ContrastResults` object

    !!! success "Credit"
        Calculations are performed by `statsmodels`.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> from ts_stat_tests.tests.linearity import linearity
        >>> x = sm.add_constant(np.linspace(0, 10, 100))
        >>> y = 3 + 2 * x[:, 1] + np.random.normal(size=100)
        >>> res = sm.OLS(y, x).fit()

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Ramsey RESET test"}
        >>> result = linearity(res, algorithm="rr")
        >>> print(f"p-value: {result.pvalue:.4f}")  # doctest: +SKIP
        p-value: 0.1367

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Rainbow test"}
        >>> stat, pval = linearity(res, algorithm="rb")
        >>> print(f"Rainbow p-value: {pval:.4f}")  # doctest: +SKIP
        Rainbow p-value: 0.5432

        ```
    """
    options: dict[str, tuple[str, ...]] = {
        "hc": ("hc", "harvey", "harvey-collier"),
        "lm": ("lm", "lagrange", "lagrange-multiplier"),
        "rb": ("rb", "rainbow"),
        "rr": ("rr", "reset", "ramsey-reset"),
    }

    # Internal helper to handle kwargs casting for ty
    def _call(
        func: Callable[..., Union[tuple[float, ...], ContrastResults]],
        **args: Union[
            float,
            int,
            str,
            bool,
            ArrayLike,
            None,
            RegressionResults,
            RegressionResultsWrapper,
        ],
    ) -> Union[tuple[float, ...], ContrastResults]:
        """
        !!! note "Summary"
            Internal helper to handle keyword arguments types.

        Params:
            func (Callable[..., Union[tuple[float, ...], ContrastResults]]):
                The function to call.
            args (Union[float, int, str, bool, ArrayLike, None, RegressionResults, RegressionResultsWrapper]):
                The arguments to pass.

        Returns:
            (Union[tuple[float, ...], ContrastResults]):
                The result of the function call.

        ???+ example "Examples"
            ```python
            # Internal use only
            ```
        """
        return func(**args)

    if algorithm in options["hc"]:
        return _call(_hc, res=res, **kwargs)

    if algorithm in options["lm"]:
        return _call(_lm, resid=res.resid, exog=res.model.exog, **kwargs)

    if algorithm in options["rb"]:
        return _call(_rb, res=res, **kwargs)

    if algorithm in options["rr"]:
        return _call(_rr, res=res, **kwargs)

    raise ValueError(
        generate_error_message(
            parameter_name="algorithm",
            value_parsed=algorithm,
            options=options,
        )
    )


@typechecked
def is_linear(
    res: Union[RegressionResults, RegressionResultsWrapper],
    algorithm: str = "rr",
    alpha: float = 0.05,
    **kwargs: Union[float, int, str, bool, ArrayLike, None],
) -> dict[str, Union[str, float, bool, None]]:
    """
    !!! note "Summary"
        Test whether the relationship in a fitted model is `linear` or not.

    ???+ abstract "Details"
        This function returns a dictionary containing the test results and a boolean indicating whether the null hypothesis of linearity is rejected at the given significance level.

    Params:
        res (Union[RegressionResults, RegressionResultsWrapper]):
            The fitted regression model to be checked.
        algorithm (str):
            Which linearity algorithm to use. See [`linearity()`][ts_stat_tests.tests.linearity.linearity] for options.
            Default: `"rr"`
        alpha (float):
            The significance level for the test.
            Default: `0.05`
        kwargs (Union[float, int, str, bool, ArrayLike, None]):
            Additional keyword arguments passed to the underlying test function.

    Returns:
        (dict[str, Union[str, float, bool, None]]):
            A dictionary with the following keys:
            - `"algorithm"` (str): The name of the algorithm used.
            - `"statistic"` (float): The test statistic.
            - `"pvalue"` (float): The p-value of the test.
            - `"alpha"` (float): The significance level used.
            - `"result"` (bool): Whether the relationship is linear (i.e., p-value > alpha).

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> import numpy as np
        >>> import statsmodels.api as sm
        >>> from ts_stat_tests.tests.linearity import is_linear
        >>> x = sm.add_constant(np.linspace(0, 10, 100))
        >>> y = 3 + 2 * x[:, 1] + np.random.normal(size=100)
        >>> res = sm.OLS(y, x).fit()

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Check linearity with RR"}
        >>> result = is_linear(res, algorithm="rr")
        >>> print(result["result"])  # doctest: +SKIP
        True

        ```
    """
    options: dict[str, tuple[str, ...]] = {
        "hc": ("hc", "harvey", "harvey-collier"),
        "lm": ("lm", "lagrange", "lagrange-multiplier"),
        "rb": ("rb", "rainbow"),
        "rr": ("rr", "reset", "ramsey-reset"),
    }

    raw_res = linearity(res=res, algorithm=algorithm, **kwargs)

    if algorithm in options["hc"] or algorithm in options["rb"]:
        # raw_res is tuple[float, float]
        stat, pvalue = raw_res[0], raw_res[1]  # type: ignore
    elif algorithm in options["lm"]:
        # raw_res is tuple[float, float, float, float]
        stat, pvalue = raw_res[0], raw_res[1]  # type: ignore
    else:
        # raw_res is ContrastResults (Ramsey RESET)
        stat = float(getattr(raw_res, "statistic", getattr(raw_res, "fvalue", np.nan)))
        pvalue = float(getattr(raw_res, "pvalue", np.nan))

    return {
        "algorithm": algorithm,
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "alpha": alpha,
        "result": bool(pvalue > alpha),
    }
