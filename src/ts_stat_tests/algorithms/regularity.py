# ============================================================================ #
#                                                                              #
#     Title: Regularity                                                        #
#     Purpose: Functions to compute regularity measures for time series data.  #
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
    This module contains algorithms to compute regularity measures for time series data, including approximate entropy, sample entropy, spectral entropy, and permutation entropy.
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
from typing import Optional, Union

# ## Python Third Party Imports ----
import numpy as np
from antropy import (
    app_entropy as a_app_entropy,
    perm_entropy as a_perm_entropy,
    sample_entropy as a_sample_entropy,
    spectral_entropy as a_spectral_entropy,
)
from numpy.typing import ArrayLike
from typeguard import typechecked


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = [
    "approx_entropy",
    "sample_entropy",
    "spectral_entropy",
    "permutation_entropy",
]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Algorithms                                                             ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@typechecked
def approx_entropy(x: ArrayLike, order: int = 2, metric: str = "chebyshev") -> float:
    return a_app_entropy(x=x, order=order, metric=metric)


@typechecked
def sample_entropy(x: ArrayLike, order: int = 2, metric: str = "chebyshev") -> float:
    return a_sample_entropy(x=x, order=order, metric=metric)


@typechecked
def permutation_entropy(
    x: ArrayLike,
    order: int = 3,
    delay: Union[int, list, np.ndarray] = 1,
    normalize: bool = False,
):
    return a_perm_entropy(x=x, order=order, delay=delay, normalize=normalize)


@typechecked
def spectral_entropy(
    x: ArrayLike,
    sf: float = 1,
    method: str = "fft",
    nperseg: Optional[int] = None,
    normalize: bool = False,
    axis: int = -1,
) -> Union[float, np.ndarray]:
    return a_spectral_entropy(
        x=x,
        sf=sf,
        method=method,
        nperseg=nperseg,
        normalize=normalize,
        axis=axis,
    )
