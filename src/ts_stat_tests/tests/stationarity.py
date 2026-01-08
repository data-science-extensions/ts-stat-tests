# ---------------------------------------------------------------------------- #
#                                                                              #
#    Setup                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Imports                                                                   ####
# ---------------------------------------------------------------------------- #


# ## Python StdLib Imports ----
from typing import Any, Union

# ## Python Third Party Imports ----
from numpy.typing import ArrayLike
from typeguard import typechecked

# ## Local First Party Imports ----
from ts_stat_tests.algorithms.stationarity import (
    adf as _adf,
    ers as _ers,
    kpss as _kpss,
    pp as _pp,
    rur as _rur,
    vr as _vr,
    za as _za,
)
from ts_stat_tests.utils.errors import generate_error_message


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["stationarity", "is_stationary"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Tests                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@typechecked
def stationarity(
    x: ArrayLike,
    algorithm: str = "adf",
    **kwargs: Any,
) -> Any:
    options: dict[str, tuple[str, ...]] = {
        "adf": ("adf", "augmented_dickey_fuller"),
        "kpss": ("kpss", "kwiatkowski_phillips_schmidt_shin"),
        "pp": ("pp", "phillips_perron"),
        "za": ("za", "zivot_andrews"),
        "ers": ("ers", "elliott_rothenberg_stock"),
        "vr": ("vr", "variance_ratio"),
        "rur": ("rur", "range_unit_root"),
    }

    if algorithm in options["adf"]:
        return _adf(x=x, **kwargs)
    if algorithm in options["kpss"]:
        return _kpss(x=x, **kwargs)
    if algorithm in options["pp"]:
        return _pp(x=x, **kwargs)
    if algorithm in options["za"]:
        return _za(x=x, **kwargs)
    if algorithm in options["ers"]:
        return _ers(y=x, **kwargs)
    if algorithm in options["vr"]:
        return _vr(y=x, **kwargs)
    if algorithm in options["rur"]:
        return _rur(x=x, **kwargs)

    raise ValueError(
        generate_error_message(
            parameter_name="algorithm",
            value_parsed=algorithm,
            options=options,
        )
    )


@typechecked
def is_stationary(
    x: ArrayLike,
    algorithm: str = "adf",
    alpha: float = 0.05,
    **kwargs: Any,
) -> dict[str, Union[str, float, bool, Any]]:
    res = stationarity(x=x, algorithm=algorithm, **kwargs)

    stat = res[0]
    pvalue_or_bool = res[1]

    # Handle H0 logic
    kpss_options = ("kpss", "kwiatkowski_phillips_schmidt_shin")

    if isinstance(pvalue_or_bool, bool):
        # Case for algorithms like 'pp' that might return a bool result directly
        is_stat = pvalue_or_bool
        pvalue = None
    else:
        pvalue = pvalue_or_bool
        if algorithm in kpss_options:
            # KPSS: Stationary if pvalue > alpha
            is_stat = pvalue > alpha
        else:
            # Others: Stationary if pvalue < alpha
            is_stat = pvalue < alpha

    return {
        "result": bool(is_stat),
        "statistic": float(stat),
        "pvalue": float(pvalue) if pvalue is not None else None,
        "alpha": float(alpha),
        "algorithm": str(algorithm),
    }
