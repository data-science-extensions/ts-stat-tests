# ============================================================================ #
#                                                                              #
#     Title: Stationarity Module                                               #
#     Purpose: Initialize the stationarity module by importing algorithms and  #
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
    This module provides tools to assess stationarity in time series data. It includes various algorithms and tests to evaluate whether a time series is stationary, meaning its statistical properties do not change over time.
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
from ts_stat_tests.stationarity.algorithms import adf, ers, kpss, pp, rur, vr, za
from ts_stat_tests.stationarity.tests import is_stationary, stationarity


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["adf", "ers", "kpss", "pp", "rur", "vr", "za", "stationarity", "is_stationary"]
