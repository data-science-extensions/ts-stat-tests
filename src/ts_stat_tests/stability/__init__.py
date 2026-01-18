# ============================================================================ #
#                                                                              #
#     Title: Stability Module                                                  #
#     Purpose: Initialize the stability module by importing algorithms and     #
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
    This module provides tools to assess the stability and lumpiness of time series data. It includes various algorithms and tests to evaluate the consistency of statistical properties over time, helping to identify periods of instability or variability in the data.
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
from ts_stat_tests.stability.algorithms import lumpiness, stability
from ts_stat_tests.stability.tests import is_lumpy, is_stable


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["stability", "lumpiness", "is_stable", "is_lumpy"]
