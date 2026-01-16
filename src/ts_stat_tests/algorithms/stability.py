# ---------------------------------------------------------------------------- #
#                                                                              #
#    Setup                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Imports                                                                   ####
# ---------------------------------------------------------------------------- #


# ## Python StdLib Imports ----
from typing import Union

# ## Python Third Party Imports ----
import numpy as np
import pandas as pd
from tsfeatures import lumpiness as ts_lumpiness, stability as ts_stability
from typeguard import typechecked


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["stability", "lumpiness"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Algorithms                                                             ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# -----------------------------------------------------------------------------#
# Stability                                                                 ####
# -----------------------------------------------------------------------------#


@typechecked
def stability(data: Union[np.ndarray, pd.DataFrame, pd.Series], freq: int = 1) -> float:
    return ts_stability(x=data, freq=freq)["stability"]


# ------------------------------------------------------------------------------#
# Lumpiness                                                                  ####
# ------------------------------------------------------------------------------#


@typechecked
def lumpiness(data: Union[np.ndarray, pd.DataFrame, pd.Series], freq: int = 1) -> float:
    return ts_lumpiness(x=data, freq=freq)["lumpiness"]
