# ============================================================================ #
#                                                                              #
#     Title: Normality Module                                                  #
#     Purpose: Initialize the normality module by importing algorithms and     #
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
    This module provides tools to assess the normality of time series data. It includes various algorithms and tests to evaluate whether a time series follows a normal distribution, which is important for many statistical analyses and modeling techniques.
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
from ts_stat_tests.normality.algorithms import ad, dp, jb, ob, sw
from ts_stat_tests.normality.tests import is_normal, normality


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["jb", "ob", "sw", "dp", "ad", "normality", "is_normal"]
