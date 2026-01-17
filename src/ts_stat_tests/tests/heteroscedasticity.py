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
import numpy as np
from numpy.typing import ArrayLike
from statsmodels.regression.linear_model import (
    RegressionResults,
    RegressionResultsWrapper,
)
from statsmodels.sandbox.stats.diagnostic import ResultsStore
from typeguard import typechecked

# ## Local First Party Imports ----
from ts_stat_tests.algorithms.heteroscedasticity import arch, bpl, gq, wlm
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
) -> Union[tuple[float, ...], ResultsStore]:
    options: dict[str, tuple[str, ...]] = {
        "arch": ("arch", "engle"),
        "bp": ("bp", "breusch-pagan", "breusch-pagan-lagrange-multiplier"),
        "gq": ("gq", "goldfeld-quandt"),
        "white": ("white",),
    }

    # Internal helper to handle kwargs casting for ty
    def _call(
        func: Callable[..., Union[tuple[float, ...], ResultsStore]],
        **args: Any,
    ) -> Union[tuple[float, ...], ResultsStore]:
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
    options: dict[str, tuple[str, ...]] = {
        "arch": ("arch", "engle"),
        "bp": ("bp", "breusch-pagan", "breusch-pagan-lagrange-multiplier"),
        "gq": ("gq", "goldfeld-quandt"),
        "white": ("white",),
    }

    raw_res = heteroscedasticity(res=res, algorithm=algorithm, **kwargs)

    if isinstance(raw_res, tuple):
        # Most statsmodels het tests return (lm, lmpval, fval, fpval) or (fval, pval)
        stat = float(raw_res[0])
        pvalue = float(raw_res[1])
    else:
        # If store=True, ResultsStore is returned
        stat = float(getattr(raw_res, "statistic", getattr(raw_res, "fvalue", np.nan)))
        pvalue = float(getattr(raw_res, "pvalue", np.nan))

    return {
        "algorithm": algorithm,
        "statistic": float(stat),
        "pvalue": float(pvalue),
        "alpha": alpha,
        "result": bool(pvalue < alpha),
    }
