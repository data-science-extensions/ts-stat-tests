# ---------------------------------------------------------------------------- #
#                                                                              #
#    Setup                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Imports                                                                   ####
# ---------------------------------------------------------------------------- #


# ## Python Third Party Imports ----
import numpy as np
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
