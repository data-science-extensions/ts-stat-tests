# ============================================================================ #
#                                                                              #
#     Title: Regularity Module                                                 #
#     Purpose: Initialize the regularity module by importing algorithms and    #
#         tests, and defining exports.                                         #
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
    This module provides tools to assess the regularity of time series data. It includes various algorithms and tests to evaluate the predictability and complexity of time series, helping to identify patterns and structures within the data.
"""


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Setup                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Imports                                                                   ####
# ---------------------------------------------------------------------------- #


# ## Local First Party Imports ----
from ts_stat_tests.regularity.algorithms import (
    approx_entropy,
    permutation_entropy,
    sample_entropy,
    spectral_entropy,
    svd_entropy,
)
from ts_stat_tests.regularity.tests import entropy, is_regular, regularity


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = [
    "approx_entropy",
    "permutation_entropy",
    "sample_entropy",
    "spectral_entropy",
    "svd_entropy",
    "entropy",
    "regularity",
    "is_regular",
]
