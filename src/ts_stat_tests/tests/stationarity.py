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
from typing import Any, Callable, Union

# ## Python Third Party Imports ----
import numpy as np
from numpy.typing import ArrayLike, NDArray
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
# Type Aliases                                                              ####
# ---------------------------------------------------------------------------- #


STATIONARITY_ITEM = Union[float, int, dict[str, float], NDArray[np.float64], ResultsStore]
"""Type alias for the items in the stationarity test result tuple."""

STATIONARITY_RETURN_TYPE = tuple[STATIONARITY_ITEM, ...]
"""Type alias for the return type of stationarity tests."""


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
) -> STATIONARITY_RETURN_TYPE:
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

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.tests.stationarity import stationarity
        >>> from ts_stat_tests.utils.data import data_normal
        >>> normal = data_normal

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Augmented Dickey-Fuller"}
        >>> result = stationarity(normal, algorithm="adf")
        >>> print(f"ADF statistic: {result[0]:.4f}")
        ADF statistic: -30.7838

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Kwiatkowski-Phillips-Schmidt-Shin"}
        >>> result = stationarity(normal, algorithm="kpss")
        >>> print(f"KPSS statistic: {result[0]:.4f}")
        KPSS statistic: 0.0858

        ```

        ```pycon {.py .python linenums="1" title="Example 3: Phillips-Perron"}
        >>> result = stationarity(normal, algorithm="pp")
        >>> print(f"PP statistic: {result[0]:.4f}")
        PP statistic: -30.7758

        ```

        ```pycon {.py .python linenums="1" title="Example 4: Zivot-Andrews"}
        >>> result = stationarity(normal, algorithm="za")
        >>> print(f"ZA statistic: {result[0]:.4f}")
        ZA statistic: -30.8800

        ```

        ```pycon {.py .python linenums="1" title="Example 5: Elliot-Rothenberg-Stock"}
        >>> result = stationarity(normal, algorithm="ers")
        >>> print(f"ERS statistic: {result[0]:.4f}")
        ERS statistic: -30.1517

        ```

        ```pycon {.py .python linenums="1" title="Example 6: Variance Ratio"}
        >>> result = stationarity(normal, algorithm="vr")
        >>> print(f"VR statistic: {result[0]:.4f}")
        VR statistic: -12.8518

        ```

        ```pycon {.py .python linenums="1" title="Example 7: Range Unit Root"}
        >>> result = stationarity(normal, algorithm="rur")
        >>> print(f"RUR statistic: {result[0]:.4f}")
        RUR statistic: 0.3479

        ```

        ```pycon {.py .python linenums="1" title="Example 8: Invalid algorithm"}
        >>> stationarity(normal, algorithm="invalid")
        Traceback (most recent call last):
            ...
        ValueError: Invalid 'algorithm': invalid. Options: {'adf': ('adf', 'augmented_dickey_fuller'), 'kpss': ('kpss', 'kwiatkowski_phillips_schmidt_shin'), 'pp': ('pp', 'phillips_perron'), 'za': ('za', 'zivot_andrews'), 'ers': ('ers', 'elliott_rothenberg_stock'), 'vr': ('vr', 'variance_ratio'), 'rur': ('rur', 'range_unit_root')}

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
    def _call(
        func: Callable[..., STATIONARITY_RETURN_TYPE],
        **args: Union[float, int, str, bool, ArrayLike, None],
    ) -> STATIONARITY_RETURN_TYPE:
        """
        !!! note "Summary"
            Internal helper to call the test function.

        Params:
            func (Callable[..., STATIONARITY_RETURN_TYPE]):
                The function to call.
            args (Union[float, int, str, bool, ArrayLike, None]):
                The arguments to pass to the function.

        Returns:
            (STATIONARITY_RETURN_TYPE):
                The result of the function call.

        ???+ example "Examples"

            ```pycon {.py .python linenums="1" title="Setup"}
            >>> from ts_stat_tests.tests.stationarity import stationarity
            >>> from ts_stat_tests.utils.data import data_normal
            >>> normal = data_normal
            ```

            ```pycon {.py .python linenums="1" title="Example 1: ADF test via internal helper"}
            >>> result = stationarity(normal, algorithm="adf")
            >>> print(f"ADF statistic: {result[0]:.4f}")
            ADF statistic: -30.7838

            ```
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
) -> dict[str, Union[str, bool, STATIONARITY_ITEM, None]]:
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

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.tests.stationarity import is_stationary
        >>> from ts_stat_tests.utils.data import data_normal
        >>> normal = data_normal

        ```

        ```pycon {.py .python linenums="1" title="Example 1: ADF test on stationary data"}
        >>> res = is_stationary(normal, algorithm="adf")
        >>> res["result"]
        True
        >>> print(f"p-value: {res['pvalue']:.4f}")
        p-value: 0.0000

        ```

        ```pycon {.py .python linenums="1" title="Example 2: KPSS test on stationary data"}
        >>> res = is_stationary(normal, algorithm="kpss")
        >>> res["result"]
        True
        >>> print(f"p-value: {res['pvalue']:.4f}")
        p-value: 0.1000

        ```

        ```pycon {.py .python linenums="1" title="Example 3: RUR test"}
        >>> res = is_stationary(normal, algorithm="rur")
        >>> res["result"]
        True
        >>> print(f"p-value: {res['pvalue']:.2f}")
        p-value: 0.01

        ```
    """
    res: Any = stationarity(x=x, algorithm=algorithm, **kwargs)

    stat: Any
    pvalue: Any

    # stationarity() always returns a tuple
    res_tuple: Any = res
    stat = res_tuple[0]
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

    # Define return dict explicitly to match return type hint
    ret: dict[str, Union[str, bool, STATIONARITY_ITEM, None]] = {
        "result": bool(is_stat),
        "statistic": float(stat) if isinstance(stat, (int, float)) else stat,
        "pvalue": float(pvalue) if pvalue is not None else None,
        "alpha": float(alpha),
        "algorithm": str(algorithm),
    }

    return ret
