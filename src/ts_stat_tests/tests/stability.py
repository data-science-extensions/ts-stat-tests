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
from typeguard import typechecked

# ## Local First Party Imports ----
from ts_stat_tests.algorithms.stability import (
    lumpiness as _lumpiness,
    stability as _stability,
)


# ---------------------------------------------------------------------------- #
# Exports                                                                   ####
# ---------------------------------------------------------------------------- #


__all__: list[str] = ["is_stable", "is_lumpy"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Algorithms                                                             ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# -----------------------------------------------------------------------------#
# Stability                                                                 ####
# -----------------------------------------------------------------------------#


@typechecked
def is_stable(data: Union[np.ndarray, pd.DataFrame, pd.Series], freq: int = 1, alpha: float = 0.5) -> bool:
    return True if _stability(data=data, freq=freq) > alpha else False


# ------------------------------------------------------------------------------#
# Lumpiness                                                                  ####
# ------------------------------------------------------------------------------#


@typechecked
def is_lumpy(data: Union[np.ndarray, pd.DataFrame, pd.Series], freq: int = 1, alpha: float = 0.5) -> bool:
    return True if _lumpiness(data=data, freq=freq) > alpha else False
