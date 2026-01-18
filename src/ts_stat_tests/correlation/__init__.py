# ============================================================================ #
#                                                                              #
#     Title: Correlation Module                                                #
#     Purpose: Correlation algorithms and tests for time series analysis.      #
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
    This module provides a suite of algorithms and tests to assess correlation in time series data. It includes functions to compute the Autocorrelation Function (ACF), Partial Autocorrelation Function (PACF), Cross-Correlation Function (CCF), and various statistical tests such as the Ljung-Box test, Lagrange Multiplier test, and Breusch-Godfrey LM test.

    The module is structured into two main submodules:
    - `algorithms`: Contains implementations of correlation algorithms.
    - `tests`: Contains statistical tests for evaluating correlation.

    Each function is designed to handle time series data efficiently, providing insights into the correlation structure of the data.
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
from ts_stat_tests.correlation.algorithms import acf, bglm, ccf, lb, lm, pacf
from ts_stat_tests.correlation.tests import correlation, is_correlated


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["acf", "pacf", "ccf", "lb", "lm", "bglm", "correlation", "is_correlated"]
