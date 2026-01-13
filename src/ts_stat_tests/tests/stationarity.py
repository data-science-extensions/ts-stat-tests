# ============================================================================ #
#                                                                              #
#     Title: Stationarity Tests                                                #
#     Purpose: Convenience functions for stationarity algorithms.              #
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
    This module contains convenience functions and tests for stationarity measures,
    allowing for easy access to different unit root and stationarity algorithms.
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
from typing import Callable, Union, cast

# ## Python Third Party Imports ----
from numpy.typing import ArrayLike
from statsmodels.stats.diagnostic import ResultsStore
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
    **kwargs: Union[float, int, str, bool, ArrayLike, None],
) -> tuple[Union[float, int, dict[str, float], ResultsStore, None], ...]:
    """
    !!! note "Summary"
        Perform a stationarity test on the given data.

    ???+ abstract "Details"
        This function is a convenience wrapper around multiple underlying algorithms:<br>
        - [`adf()`][ts_stat_tests.algorithms.stationarity.adf]<br>
        - [`kpss()`][ts_stat_tests.algorithms.stationarity.kpss]<br>
        - [`pp()`][ts_stat_tests.algorithms.stationarity.pp]<br>
        - [`za()`][ts_stat_tests.algorithms.stationarity.za]<br>
        - [`ers()`][ts_stat_tests.algorithms.stationarity.ers]<br>
        - [`vr()`][ts_stat_tests.algorithms.stationarity.vr]<br>
        - [`rur()`][ts_stat_tests.algorithms.stationarity.rur]

    Params:
        x (ArrayLike):
            The data to be checked.
        algorithm (str):
            Which stationarity algorithm to use.<br>
            - `adf()`: `["adf", "augmented_dickey_fuller"]`<br>
            - `kpss()`: `["kpss", "kwiatkowski_phillips_schmidt_shin"]`<br>
            - `pp()`: `["pp", "phillips_perron"]`<br>
            - `za()`: `["za", "zivot_andrews"]`<br>
            - `ers()`: `["ers", "elliott_rothenberg_stock"]`<br>
            - `vr()`: `["vr", "variance_ratio"]`<br>
            - `rur()`: `["rur", "range_unit_root"]`<br>
            Defaults to `"adf"`.
        kwargs (Union[float, int, str, bool, ArrayLike, None]):
            Additional arguments to pass to the underlying algorithm.

    Raises:
        (ValueError):
            When the given value for `algorithm` is not valid.

    Returns:
        (tuple[Union[float, int, dict, ResultsStore, None], ...]):
            The result of the stationarity test.

    !!! success "Credit"
        Calculations are performed by `statsmodels`, `arch`, and `pmdarima`.

    ???+ example "Examples"

        `stationarity` with `augmented_dickey_fuller` algorithm:
        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> import numpy as np
        >>> from ts_stat_tests.tests.stationarity import stationarity
        >>> data = np.random.normal(0, 1, 100)
        >>> result = stationarity(data, algorithm="adf")
        ```
    """
    options: dict[str, tuple[str, ...]] = {
        "adf": ("adf", "augmented_dickey_fuller"),
        "kpss": ("kpss", "kwiatkowski_phillips_schmidt_shin"),
        "pp": ("pp", "phillips_perron"),
        "za": ("za", "zivot_andrews"),
        "ers": ("ers", "elliott_rothenberg_stock"),
        "vr": ("vr", "variance_ratio"),
        "rur": ("rur", "range_unit_root"),
    }

    # Internal helper to handle kwargs casting for ty
    def _call(func: Any, **args: Any) -> Any:
        """
        !!! note "Summary"
            Internal helper to call the test function.
        """
        return func(**args)

    if algorithm in options["adf"]:
        return _call(_adf, x=x, **kwargs)
    if algorithm in options["kpss"]:
        return _call(_kpss, x=x, **kwargs)
    if algorithm in options["pp"]:
        return _call(_pp, x=x, **kwargs)
    if algorithm in options["za"]:
        return _call(_za, x=x, **kwargs)
    if algorithm in options["ers"]:
        return _call(_ers, y=x, **kwargs)
    if algorithm in options["vr"]:
        return _call(_vr, y=x, **kwargs)
    if algorithm in options["rur"]:
        return _call(_rur, x=x, **kwargs)

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
    **kwargs: Union[float, int, str, bool, ArrayLike, None],
) -> dict[str, Union[str, float, bool, None]]:
    """
    !!! note "Summary"
        Test whether a given data set is `stationary` or not.

    ???+ abstract "Details"
        This function checks the results of a stationarity test against a significance level `alpha`.

        Note that different tests have different null hypotheses:
        - For ADF, PP, ZA, ERS, VR, RUR: H0 is non-stationarity (unit root). Stationary if p-value < alpha.
        - For KPSS: H0 is stationarity. Stationary if p-value > alpha.

    Params:
        x (ArrayLike):
            The data to be checked.
        algorithm (str):
            Which stationarity algorithm to use. Defaults to `"adf"`.
        alpha (float, optional):
            The significance level for the test. Defaults to `0.05`.
        kwargs (Union[float, int, str, bool, ArrayLike, None]):
            Additional arguments to pass to the underlying algorithm.

    Returns:
        (dict[str, Union[str, float, bool, None]]):
            A dictionary containing:
            - `"result"` (bool): Indicator if the series is stationary.
            - `"statistic"` (float): The test statistic.
            - `"pvalue"` (float): The p-value of the test.
            - `"alpha"` (float): The significance level used.
            - `"algorithm"` (str): The algorithm used.

    ???+ example "Examples"

        `is_stationary` with `adf` algorithm:
        ```pycon {.py .python linenums="1" title="Basic usage"}
        >>> import numpy as np
        >>> from ts_stat_tests.tests.stationarity import is_stationary
        >>> data = np.random.normal(0, 1, 100)
        >>> result = is_stationary(data, algorithm="adf")
        >>> result["result"]
        True
        ```
    """
    res = stationarity(x=x, algorithm=algorithm, **kwargs)

    stat: Union[float, int, dict[str, float]]
    pvalue: Union[float, int, dict[str, float], bool, str, None]

    # stationarity() always returns a tuple
    res_tuple = res
    stat = cast(Union[float, int, dict[str, float]], res_tuple[0])
    pvalue_or_bool = res_tuple[1]

    # Handle H0 logic
    stationary_h0 = (
        "kpss",
        "kwiatkowski_phillips_schmidt_shin",
    )

    is_stat: bool = False
    pvalue = None

    if isinstance(pvalue_or_bool, bool):
        is_stat = pvalue_or_bool
    elif isinstance(pvalue_or_bool, (int, float)):
        pvalue = pvalue_or_bool
        if algorithm in stationary_h0:
            is_stat = bool(pvalue > alpha)
        else:
            is_stat = bool(pvalue < alpha)

    return cast(
        dict[str, Union[str, float, bool, None]],
        {
            "result": bool(is_stat),
            "statistic": float(stat) if isinstance(stat, (int, float)) else stat,  # type: ignore
            "pvalue": float(pvalue) if isinstance(pvalue, (int, float)) else pvalue,  # type: ignore
            "alpha": float(alpha),
            "algorithm": str(algorithm),
        },
    )
