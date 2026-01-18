# ============================================================================ #
#                                                                              #
#     Title: Linearity Module                                                  #
#     Purpose: Initialize the linearity module by importing algorithms and     #
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
    This module provides tools to assess the linearity of time series data. It includes various algorithms and tests to evaluate whether a time series exhibits linear behavior, helping to identify relationships and trends within the data.
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
from ts_stat_tests.linearity.algorithms import hc, lm, rb, rr
from ts_stat_tests.linearity.tests import is_linear, linearity


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["hc", "lm", "rb", "rr", "linearity", "is_linear"]
