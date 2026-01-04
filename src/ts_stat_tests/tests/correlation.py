# ---------------------------------------------------------------------------- #
#                                                                              #
#    Setup                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Imports                                                                   ####
# ---------------------------------------------------------------------------- #


# ## Python StdLib Imports ----
from typing import Union

# ## Python Third Party Imports ----
import numpy as np
from numpy.typing import ArrayLike
from typeguard import typechecked

# ## Local First Party Imports ----
from ts_stat_tests.algorithms.correlation import (
    acf as _acf,
    bglm as _bglm,
    ccf as _ccf,
    lb as _lb,
    lm as _lm,
    pacf as _pacf,
)
from ts_stat_tests.utils.errors import generate_error_message


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["correlation", "is_correlated"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Tests                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


def correlation(x: ArrayLike, algorithm: str, **kwargs) -> Union[np.ndarray, tuple[np.ndarray, ...]]:
    options: dict[str, tuple[str, ...]] = {
        "acf": ("acf", "auto", "ac"),
        "pacf": ("pacf", "partial", "pc"),
        "alb": ("alb", "acorr_ljungbox", "acor_lb", "a_lb", "lb", "ljungbox"),
        "alm": ("alm", "acorr_lm", "a_lm", "lm"),
        "ccf": ("ccf", "cross", "cross_correlation", "cross-correlation", "cc"),
        "bglm": ("bglm", "breusch_godfrey", "bg"),
    }
    # options_r = dict_reverse_keys_and_values(options)
    # if algorithm in options_r.keys():
    #     return locals()[options_r[algorithm]](x=x, **kwargs)
    if algorithm in options["acf"]:
        return _acf(x=x, **kwargs)
    elif algorithm in options["pacf"]:
        return _pacf(x=x, **kwargs)
    elif algorithm in options["alb"]:
        return _lb(x=x, **kwargs)
    elif algorithm in options["alm"]:
        return _lm(resid=x, **kwargs)
    elif algorithm in options["ccf"]:
        if "y" not in kwargs or kwargs["y"] is None:
            raise ValueError("y must be provided for ccf algorithm")
        return _ccf(x=x, **kwargs)
    elif algorithm in options["bglm"]:
        return _bglm(resid=x, **kwargs)
    else:
        raise ValueError(
            generate_error_message(
                parameter_name="algorithm",
                value_parsed=algorithm,
                options=options,
            )
        )


@typechecked
def is_correlated() -> None:
    """
    !!! note "Summary"
        A placeholder function for checking if a time series is correlated.
    """
    return None
