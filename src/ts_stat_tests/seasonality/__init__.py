# ============================================================================ #
#                                                                              #
#     Title: Seasonality Module                                                #
#     Purpose: Initialize the seasonality module by importing algorithms and   #
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
    This module provides tools to assess seasonality in time series data. It includes various algorithms and tests to evaluate the presence and strength of seasonal patterns, helping to identify recurring trends and cycles within the data.
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
from ts_stat_tests.seasonality.algorithms import (
    ch,
    ocsb,
    qs,
    seasonal_strength,
    spikiness,
    trend_strength,
)
from ts_stat_tests.seasonality.tests import is_seasonal, seasonality


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = [
    "ch",
    "ocsb",
    "qs",
    "seasonal_strength",
    "spikiness",
    "trend_strength",
    "seasonality",
    "is_seasonal",
]
