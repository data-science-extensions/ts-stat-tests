# ============================================================================ #
#                                                                              #
#     Title: Error Utilities                                                   #
#     Purpose: Functions for generating standardized error messages and        #
#         performing data equality checks.                                     #
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
    This module contains utility functions for generating standardized error messages and performing data equality checks.
"""


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Setup                                                                 ####
#                                                                              #
# ---------------------------------------------------------------------------- #


## --------------------------------------------------------------------------- #
##  Imports                                                                 ####
## --------------------------------------------------------------------------- #


# ## Python StdLib Imports ----
from collections.abc import Collection, Mapping
from typing import Optional, Union, overload

# ## Python Third Party Imports ----
from typeguard import typechecked


## --------------------------------------------------------------------------- #
##  Constants                                                               ####
## --------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Functions                                                             ####
#                                                                              #
# ---------------------------------------------------------------------------- #


## --------------------------------------------------------------------------- #
##  Error messages                                                          ####
## --------------------------------------------------------------------------- #


@typechecked
def generate_error_message(
    parameter_name: str,
    value_parsed: object,
    options: Union[
        Mapping[str, str],
        Mapping[str, Collection[Union[str, int, float]]],
    ],
) -> str:
    """
    !!! note "Summary"
        Generates a standardised error message for invalid parameter values.

    Params:
        parameter_name (str):
            The name of the parameter.
        value_parsed (object):
            The invalid value provided.
        options (Union[Mapping[str, str], Mapping[str, Collection[Union[str, int, float]]]]):
            A dictionary mapping valid option keys to their acceptable values.

    Returns:
        (str):
            A formatted error message listing valid options.
    """
    error_message: str = f"Invalid option for `{parameter_name}` parameter: '{value_parsed}'.\n"
    for key, value in options.items():
        error_message += f"For the '{key}' option, use one of: '{value}'.\n"
    return error_message


## --------------------------------------------------------------------------- #
##  Data equality                                                           ####
## --------------------------------------------------------------------------- #


@overload
def is_almost_equal(first: float, second: float, *, places: int = 7) -> bool: ...
@overload
def is_almost_equal(first: float, second: float, *, delta: float) -> bool: ...
@typechecked
def is_almost_equal(
    first: float, second: float, *, places: Optional[int] = None, delta: Optional[float] = None
) -> bool:
    """
    !!! note "Summary"
        Checks if two float values are almost equal within a specified precision.

    Params:
        first (float):
            The first float value.
        second (float):
            The second float value.
        places (Optional[int]):
            The number of decimal places for comparison. Defaults to 7 if not provided.
        delta (Optional[float]):
            An optional delta value for comparison.

    Returns:
        (bool):
            `True` if the values are almost equal, `False` otherwise.

    !!! success "Credit"
        Inspiration from Python's UnitTest function [`assertAlmostEqual`](https://github.com/python/cpython/blob/3.11/Lib/unittest/case.py).
    """
    if places is not None and delta is not None:
        raise ValueError(f"Specify `delta` or `places`, not both.")
    if first == second:
        return True
    diff: float = abs(first - second)
    if delta is not None:
        if diff <= delta:
            return True
    else:
        places = places or 7
        if round(diff, places) == 0:
            return True
    return False


@overload
def assert_almost_equal(first: float, second: float, msg: Optional[str] = None, *, places: int = 7) -> None: ...
@overload
def assert_almost_equal(first: float, second: float, msg: Optional[str] = None, *, delta: float) -> None: ...
@typechecked
def assert_almost_equal(
    first: float,
    second: float,
    msg: Optional[str] = None,
    *,
    places: Optional[int] = None,
    delta: Optional[float] = None,
) -> None:
    """
    !!! note "Summary"
        Asserts that two float values are almost equal within a specified precision.

    Params:
        first (float):
            The first float value.
        second (float):
            The second float value.
        places (Optional[int]):
            The number of decimal places for comparison. Defaults to 7 if not provided.
        msg (Optional[str]):
            An optional message to include in the exception if the values are not almost equal.
        delta (Optional[float]):
            An optional delta value for comparison.

    Returns:
        (None):
            None. Raises an `AssertionError` if the values are not almost equal to within the tolerances specified.

    !!! success "Credit"
        Inspiration from Python's UnitTest function [`assertAlmostEqual`](https://github.com/python/cpython/blob/3.11/Lib/unittest/case.py).
    """
    params: dict[str, Optional[Union[float, int]]] = {
        "first": first,
        "second": second,
        "places": places,
        "delta": delta,
    }
    if is_almost_equal(**{k: v for k, v in params.items() if v is not None}):
        return None
    diff: float = abs(first - second)
    if delta is not None:
        standard_msg: str = f"{first} != {second} within {delta} delta ({diff} difference)"
    else:
        places = places or 7
        standard_msg = f"{first} != {second} within {places} places ({diff} difference)"
    msg = msg or standard_msg
    raise AssertionError(msg)
