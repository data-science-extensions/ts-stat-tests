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
from statsmodels import api as sm
from statsmodels.stats.api import (
    acorr_breusch_godfrey,
    acorr_ljungbox,
    acorr_lm,
)
from statsmodels.tsa.stattools import (
    acf as st_acf,
    ccf as st_ccf,
    pacf as st_pacf,
)

# ## Local First Party Imports ----
from tests.setup import BaseTester
from ts_stat_tests.algorithms.correlation import acf, bglm, ccf, lb, lm, pacf
from ts_stat_tests.tests.correlation import correlation, is_correlated
from ts_stat_tests.utils.data import get_uniform_data, load_airline, load_macrodata
from ts_stat_tests.utils.errors import assert_almost_equal, is_almost_equal


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Testing                                                                ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Tests                                                                     ####
# ---------------------------------------------------------------------------- #


class TestCorrelation(BaseTester):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.result_acf = acf(cls.data_airline)
        cls.result_pacf = pacf(cls.data_airline)
        cls.result_ccf = ccf(cls.data_airline, np.array(cls.data_airline) + 1)
        cls.result_alb = lb(cls.data_airline)
        cls.result_alm = lm(cls.data_airline)
        y = sm.datasets.longley.load_pandas().endog
        X = sm.datasets.longley.load_pandas().exog
        X = sm.add_constant(X)
        cls.result_abg = bglm(sm.OLS(y, X).fit())

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_acf_results(self) -> None:
        np.testing.assert_almost_equal(
            self.result_acf,
            st_acf(self.data_airline),
        )

    def test_pacf_results(self) -> None:
        np.testing.assert_almost_equal(
            self.result_pacf,
            st_pacf(self.data_airline),
        )

    def test_ccf_results(self) -> None:
        np.testing.assert_almost_equal(
            self.result_ccf,
            st_ccf(self.data_airline, np.array(self.data_airline) + 1),
        )

    def test_alb_results(self) -> None:
        np.testing.assert_almost_equal(
            np.array(self.result_alb),
            np.array(acorr_ljungbox(self.data_airline)),
        )

    def test_alm_results(self) -> None:
        np.testing.assert_almost_equal(
            self.result_alm,
            acorr_lm(self.data_airline),
        )

    def test_abg_results(self) -> None:
        y = sm.datasets.longley.load_pandas().endog
        X = sm.datasets.longley.load_pandas().exog
        X = sm.add_constant(X)
        np.testing.assert_almost_equal(
            self.result_abg,
            acorr_breusch_godfrey(sm.OLS(y, X).fit()),
        )

    def test_correlation_acf(self) -> None:
        np.testing.assert_almost_equal(
            correlation(x=self.data_airline, algorithm="acf"),
            self.result_acf,
        )

    def test_correlation_pacf(self) -> None:
        np.testing.assert_almost_equal(
            correlation(x=self.data_airline, algorithm="pacf"),
            self.result_pacf,
        )

    def test_correlation_alb(self) -> None:
        np.testing.assert_almost_equal(
            np.array(correlation(x=self.data_airline, algorithm="alb")),
            np.array(self.result_alb),
        )

    def test_correlation_alm(self) -> None:
        np.testing.assert_almost_equal(
            correlation(x=self.data_airline, algorithm="alm"),
            self.result_alm,
        )

    def test_correlation_ccf(self) -> None:
        np.testing.assert_almost_equal(
            self.result_ccf,
            correlation(
                x=self.data_airline,
                y=np.array(self.data_airline) + 1,
                algorithm="ccf",
            ),
        )

    def test_correlation_raises(self) -> None:
        with raises(ValueError):
            correlation(x=self.data_airline, algorithm="xxxx")

    def test_correlation_ccf_raises(self) -> None:
        with raises(ValueError):
            correlation(x=self.data_airline, algorithm="ccf")

    def test_correlation_bglm(self) -> None:
        y = sm.datasets.longley.load_pandas().endog
        X = sm.datasets.longley.load_pandas().exog
        X = sm.add_constant(X)
        res = sm.OLS(y, X).fit()
        np.testing.assert_almost_equal(
            correlation(res, algorithm="bglm"),
            self.result_abg,
        )

    def test_is_correlated(self) -> None:
        with raises(NotImplementedError):
            is_correlated()

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
            with raises(TypeError, match="Expected a pandas DataFrame from the data source."):
                load_macrodata()
        load_macrodata.cache_clear()

    def test_get_uniform_data(self) -> None:
        res = get_uniform_data(42)
        assert isinstance(res, np.ndarray)
        assert res.shape == (1000,)

    def test_is_almost_equal_value_error(self) -> None:
        with raises(ValueError, match="Specify `delta` or `places`, not both."):
            is_almost_equal(1.0, 1.1, places=7, delta=0.1)

    def test_is_almost_equal_with_delta(self) -> None:
        assert is_almost_equal(1.0, 1.05, delta=0.1) is True
        assert is_almost_equal(1.0, 1.2, delta=0.1) is False

    def test_assert_almost_equal_failures(self) -> None:
        with raises(AssertionError, match="within 0.1 delta"):
            assert_almost_equal(1.0, 1.2, delta=0.1)

        with raises(AssertionError, match="within 10 places"):
            assert_almost_equal(1.0, 1.0000001, places=10)

        with raises(AssertionError, match="Custom error message"):
            assert_almost_equal(1.0, 2.0, msg="Custom error message")
