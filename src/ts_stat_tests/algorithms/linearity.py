# ---------------------------------------------------------------------------- #
#                                                                              #
#    Setup                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Imports                                                                   ####
# ---------------------------------------------------------------------------- #


# ## Python StdLib Imports ----
from typing import Any, Callable, Literal, Optional, Union

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
    res_hc: Any = linear_harvey_collier(res=res, order_by=order_by, skip=skip)
    return float(res_hc.statistic), float(res_hc.pvalue)


@typechecked
def lm(
    resid: NDArray[np.float64], exog: NDArray[np.float64], func: Optional[Callable] = None
) -> tuple[float, float, float, float]:
    res_lm: Any = linear_lm(resid=resid, exog=exog, func=func)
    return float(res_lm[0]), float(res_lm[1]), float(res_lm[2].fvalue), float(res_lm[2].pvalue)


@typechecked
def rb(
    res: Union[RegressionResults, RegressionResultsWrapper],
    frac: float = 0.5,
    order_by: Optional[Union[ArrayLike, str, list[str]]] = None,
    use_distance: bool = False,
    center: Optional[Union[float, int]] = None,
) -> tuple[float, float]:
    res_rb: Any = linear_rainbow(res=res, frac=frac, order_by=order_by, use_distance=use_distance, center=center)
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
    _power: Any = power
    return linear_reset(
        res=res,
        power=_power,
        test_type=test_type,
        use_f=use_f,
        cov_type=cov_type,
        cov_kwargs=cov_kwargs,
    )
