# ============================================================================ #
#                                                                              #
#     Title: Heteroscedasticity Module                                         #
#     Purpose: Initialize the heteroscedasticity module by importing           #
#         algorithms and tests, and defining exports.                          #
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
    This module provides tools to assess heteroscedasticity in time series data. It includes various algorithms and tests to evaluate the presence of heteroscedasticity, which is the condition where the variance of errors differs across observations in a dataset.
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
from ts_stat_tests.heteroscedasticity.algorithms import arch, bpl, gq, wlm
from ts_stat_tests.heteroscedasticity.tests import (
    heteroscedasticity,
    is_heteroscedastic,
)


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["arch", "bpl", "gq", "wlm", "heteroscedasticity", "is_heteroscedastic"]
