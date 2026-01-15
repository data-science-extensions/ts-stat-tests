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

    Provides utility functions for generating standardized error messages and performing numeric assertions.

    This module includes functions to format error messages consistently and check if numeric values are within a specified tolerance, which is useful for testing and validation purposes.
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
from typing import Mapping, Optional, Union, overload

# ## Python Third Party Imports ----
from typeguard import typechecked


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
    value_parsed: str,
    options: Union[Mapping[str, Union[tuple[str, ...], list[str], str]], tuple[str, ...], list[str]],
) -> str:
    r"""
    !!! note "Summary"
        Generates a formatted error message for mismatched values or invalid options.

    ???+ abstract "Details"
        This function constructs a standardized string that describes a mismatch between a provided value and the allowed options for a given parameter. It is primarily used to provide clear, consistent feedback in `ValueError` exceptions within dispatchers.

    Params:
        parameter_name (str):
            The name of the parameter or variable being checked.
        value_parsed (str):
            The actual value that was received.
        options (Union[Mapping[str, Union[tuple[str, ...], list[str], str]], tuple[str, ...], list[str]]):
            The set of valid options or a dictionary mapping categories to valid options.

    Returns:
        (str):
            A formatted error message string.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.utils.errors import generate_error_message

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Simple Options"}
        >>> msg = generate_error_message("param", "invalid", ["opt1", "opt2"])
        >>> print(msg)
        Invalid 'param': invalid. Options: ['opt1', 'opt2']

        ```

    ??? question "References"
        1. [Python F-Strings documentation](https://docs.python.org/3/reference/lexical_analysis.html#f-strings)

    """
    return f"Invalid '{parameter_name}': {value_parsed}. Options: {options}"


@overload
def is_almost_equal(first: float, second: float, *, places: int = 7) -> bool: ...
@overload
def is_almost_equal(first: float, second: float, *, delta: float) -> bool: ...
@typechecked
def is_almost_equal(
    first: float,
    second: float,
    *,
    places: Optional[int] = None,
    delta: Optional[float] = None,
) -> bool:
    r"""
    !!! note "Summary"
        Checks if two float values are almost equal within a specified precision.

    ???+ abstract "Details"
        Determines the equality of two floating-point numbers within a tolerance. This is necessary because floating-point arithmetic can introduce small errors that make direct equality comparisons (e.g., `a == b`) unreliable.

        The user can specify tolerance either by `places` (decimal places) or by an absolute `delta`.

    Params:
        first (float):
            The first float value.
        second (float):
            The second float value.
        places (Optional[int]):
            The number of decimal places for comparison. Defaults to 7 if not provided.
        delta (Optional[float]):
            An optional delta value for comparison.

    Raises:
        (ValueError):
            If both `places` and `delta` are provided.

    Returns:
        (bool):
            `True` if the values are almost equal, `False` otherwise.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.utils.errors import is_almost_equal

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Using `places`"}
        >>> res_places = is_almost_equal(1.0, 1.00000001, places=7)
        >>> print(res_places)
        True

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Using `delta`"}
        >>> res_delta = is_almost_equal(1.0, 1.1, delta=0.2)
        >>> print(res_delta)
        True

        ```

        ```pycon {.py .python linenums="1" title="Example 3: Not Almost Equal"}
        >>> res_not_equal = is_almost_equal(1.0, 1.1, places=3)
        >>> print(res_not_equal)
        False

        ```

    ??? equation "Calculation"

        The comparison depends on whether `delta` or `places` is provided.

        If `delta` is specified:

        $$
        |first - second| \le \text{delta}
        $$

        If `places` is specified (defaults to 7):

        $$
        \text{round}(|first - second|, \text{places}) = 0
        $$

    ??? success "Credit"
        Inspiration from Python's UnitTest function [`assertAlmostEqual`](https://github.com/python/cpython/blob/3.11/Lib/unittest/case.py).

    ??? question "References"
        1. [Python unittest source code](https://github.com/python/cpython/blob/main/Lib/unittest/case.py)

    """
    if places is not None and delta is not None:
        raise ValueError("Specify `delta` or `places`, not both.")
    if first == second:
        return True
    diff: float = abs(first - second)
    if delta is not None:
        if diff <= delta:
            return True
    else:
        places_val: int = places if places is not None else 7
        if round(diff, places_val) == 0:
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
    r"""
    !!! note "Summary"
        Asserts that two float values are almost equal within a specified precision.

    ???+ abstract "Details"
        Performs a floating-point comparison using [is_almost_equal][ts_stat_tests.utils.errors.is_almost_equal]. If the comparison fails, an `AssertionError` is raised with a descriptive message.

    Params:
        first (float):
            The first float value.
        second (float):
            The second float value.
        msg (Optional[str]):
            An optional message to include in the exception if the values are not almost equal.
        places (Optional[int]):
            The number of decimal places for comparison. Defaults to 7 if not provided.
        delta (Optional[float]):
            An optional delta value for comparison.

    Raises:
        (ValueError):
            If both `places` and `delta` are provided.
        (AssertionError):
            If the two float values are not almost equal.

    Returns:
        (None):
            None. Raises an `AssertionError` if the values are not almost equal to within the tolerances specified.

    ???+ example "Examples"

        ```pycon {.py .python linenums="1" title="Setup"}
        >>> from ts_stat_tests.utils.errors import assert_almost_equal

        ```

        ```pycon {.py .python linenums="1" title="Example 1: Using `places`"}
        >>> res_places = assert_almost_equal(1.0, 1.0, places=7)
        >>> print(res_places is None)
        True

        ```

        ```pycon {.py .python linenums="1" title="Example 2: Using `delta`"}
        >>> res_delta = assert_almost_equal(1.0, 1.1, delta=0.2)
        >>> print(res_delta is None)
        True

        ```

        ```pycon {.py .python linenums="1" title="Example 3: AssertionError Raised"}
        >>> assert_almost_equal(1.0, 1.1, places=3)
        Traceback (most recent call last):
            ...
        AssertionError: Assertion failed: 1.0 != 1.1 (places=3, delta=None)

        ```

    ??? equation "Calculation"

        Refer to [is_almost_equal][ts_stat_tests.utils.errors.is_almost_equal] for the underlying logic.

    ??? success "Credit"
        Inspiration from Python's UnitTest function [`assertAlmostEqual`](https://github.com/python/cpython/blob/3.11/Lib/unittest/case.py).

    ??? question "References"
        1. [Python unittest source code](https://github.com/python/cpython/blob/main/Lib/unittest/case.py)

    """
    is_equal: bool = False
    if delta is not None:
        is_equal = is_almost_equal(first, second, delta=delta)
    else:
        places_val: int = places if places is not None else 7
        is_equal = is_almost_equal(first, second, places=places_val)

    if not is_equal:
        error_msg: str = msg if msg is not None else f"Assertion failed: {first} != {second} ({places=}, {delta=})"
        raise AssertionError(error_msg)
