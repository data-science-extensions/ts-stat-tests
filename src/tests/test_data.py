# ---------------------------------------------------------------------------- #
#                                                                              #
#    Setup                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Imports                                                                   ####
# ---------------------------------------------------------------------------- #


# ## Python StdLib Imports ----
from unittest.mock import patch

# ## Python Third Party Imports ----
import numpy as np
import pandas as pd
from pytest import raises
from typeguard import TypeCheckError

# ## Local First Party Imports ----
from tests.setup import BaseTester
from ts_stat_tests.utils.data import get_uniform_data, load_airline, load_macrodata


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Testing                                                                ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Tests                                                                     ####
# ---------------------------------------------------------------------------- #


class TestData(BaseTester):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_load_airline_type_error(self) -> None:
        mock_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        load_airline.cache_clear()
        with patch("pandas.read_csv", return_value=mock_df):
            with raises(TypeError, match="Expected a pandas Series from the data source."):
                load_airline()
        load_airline.cache_clear()

    def test_load_macrodata_type_error(self) -> None:
        load_macrodata.cache_clear()
        with patch("pandas.read_csv", return_value=pd.Series(dtype=float)):
            with raises(TypeCheckError) as e:
                load_macrodata()
            assert "value assigned to data (pandas.core.series.Series)" in str(e.value)
            assert "is not an instance of" in str(e.value)
            assert "pandas.core.frame.DataFrame" in str(e.value)
        load_macrodata.cache_clear()

    def test_get_uniform_data(self) -> None:
        res = get_uniform_data(42)
        assert isinstance(res, np.ndarray)
        assert res.shape == (1000,)
