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
from typing import Literal, Optional, Union

# ## Python Third Party Imports ----
import numpy as np
from antropy import (
    app_entropy as a_app_entropy,
    perm_entropy as a_perm_entropy,
    sample_entropy as a_sample_entropy,
    spectral_entropy as a_spectral_entropy,
    svd_entropy as a_svd_entropy,
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
    "svd_entropy",
]


## --------------------------------------------------------------------------- #
##  Constants                                                               ####
## --------------------------------------------------------------------------- #


VALID_KDTREE_METRIC_OPTIONS = Literal[
    "euclidean", "l2", "minkowski", "p", "manhattan", "cityblock", "l1", "chebyshev", "infinity"
]
# from sklearn.neighbors import KDTree; print(KDTree.valid_metrics);


VALID_SPECTRAL_ENTROPY_METHOD_OPTIONS = Literal["fft", "welch"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Algorithms                                                             ####
#                                                                              #
# ---------------------------------------------------------------------------- #


@typechecked
def approx_entropy(
    x: ArrayLike,
    order: int = 2,
    tolerance: Optional[float] = None,
    metric: VALID_KDTREE_METRIC_OPTIONS = "chebyshev",
) -> float:
    return a_app_entropy(
        x=x,
        order=order,
        tolerance=tolerance,
        metric=metric,
    )


@typechecked
def sample_entropy(
    x: ArrayLike,
    order: int = 2,
    tolerance: Optional[float] = None,
    metric: VALID_KDTREE_METRIC_OPTIONS = "chebyshev",
) -> float:
    return a_sample_entropy(
        x=x,
        order=order,
        tolerance=tolerance,
        metric=metric,
    )


@typechecked
def permutation_entropy(
    x: ArrayLike,
    order: int = 3,
    delay: Union[int, list, np.ndarray] = 1,
    normalize: bool = False,
) -> float:
    return a_perm_entropy(
        x=x,
        order=order,
        delay=delay,
        normalize=normalize,
    )


@typechecked
def spectral_entropy(
    x: ArrayLike,
    sf: float = 1,
    method: VALID_SPECTRAL_ENTROPY_METHOD_OPTIONS = "fft",
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


@typechecked
def svd_entropy(
    x: ArrayLike,
    order: int = 3,
    delay: int = 1,
    normalize: bool = False,
) -> float:
    return a_svd_entropy(
        x=x,
        order=order,
        delay=delay,
        normalize=normalize,
    )
