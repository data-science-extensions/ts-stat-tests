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
import pandas as pd
from pmdarima.arima import ARIMA
from pytest import raises

# ## Local First Party Imports ----
from tests.setup import BaseTester
from ts_stat_tests.algorithms.seasonality import (
    ch,
    ocsb,
    qs,
    seasonal_strength,
    spikiness,
    trend_strength,
)
from ts_stat_tests.tests.seasonality import is_seasonal, seasonality
from ts_stat_tests.utils.errors import assert_almost_equal


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Testing                                                                ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Tests                                                                     ####
# ---------------------------------------------------------------------------- #


class TestSeasonality(BaseTester):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.result_qs_simple = qs(x=cls.data_airline, freq=12)
        cls.result_qs_complex = qs(x=cls.data_airline, freq=12, residuals=True)
        cls.result_ocsb = ocsb(x=cls.data_airline, m=12)
        cls.result_ch = ch(x=cls.data_airline, m=12)
        cls.result_seasonal_strength = seasonal_strength(x=cls.data_airline, m=12)
        cls.result_trend_strength = trend_strength(x=cls.data_airline, m=12)
        cls.result_spikiness = spikiness(x=cls.data_airline, m=12)

    def setUp(self) -> None:
        pass

    def test_qs_defaults(self) -> None:
        assert isinstance(self.result_qs_simple, tuple)
        assert isinstance(self.result_qs_simple[0], float)
        assert isinstance(self.result_qs_simple[1], float)
        assert isinstance(self.result_qs_complex[0], float)
        assert isinstance(self.result_qs_complex[1], float)
        assert isinstance(self.result_qs_complex[2], (dict, type(None), ARIMA))

    def test_qs(self) -> None:

        # To get the full list of necessary permutations, run:
        # >>> import itertools
        # >>> print(list(itertools.product(*[[True,False]]*3)))

        params = [
            (True, True, True, 101.8592939, 7.6126411e-23),
            (True, True, False, 131.4499985, 2.8575608e-29),
            (True, False, True, 194.4692892, 5.9092232e-43),
            (True, False, False, 194.4692892, 5.9092232e-43),
            (False, True, True, 132.5823829, 1.6221885e-29),
            (False, True, False, 145.1209623, 3.0717326e-32),
            (False, False, True, 141.7127876, 1.6883370e-31),
            (False, False, False, 141.7127876, 1.6883370e-31),
        ]

        for diff, residuals, autoarima, expected_stat, expected_pval in params:
            qs_result = qs(
                x=self.data_airline,
                freq=12,
                diff=diff,
                residuals=residuals,
                autoarima=autoarima,
            )
            assert_almost_equal(qs_result[0], expected_stat, places=5)
            assert_almost_equal(qs_result[1], expected_pval, places=5)
            if residuals:
                assert isinstance(qs_result[2], ARIMA)

    def test_qs_failures(self) -> None:
        with raises(AttributeError):
            qs(pd.Series([None, None]), 12)
        with raises(AttributeError):
            qs(self.data_airline, 1)
        with raises(ValueError):
            qs(pd.Series([1, 1]), 12, True, True, False)
        with raises(ValueError):
            qs(pd.Series([0, 1]), 4, True, True, True)

    def test_ocsb(self) -> None:
        assert self.result_ocsb == 1

    def test_ch(self) -> None:
        assert self.result_ch == 0

    def test_seasonal_strength(self) -> None:
        assert_almost_equal(self.result_seasonal_strength, 0.7787219427520644, places=7)
        assert seasonal_strength(np.sin(2 * np.pi * 1 * np.arange(3000) / 100), 100) == 1

    def test_trend_strength(self) -> None:
        assert_almost_equal(self.result_trend_strength, 0.965679739099362, places=7)
        assert trend_strength(np.arange(1000), 1) == 1

    def test_spikiness(self) -> None:
        assert_almost_equal(self.result_spikiness, 0.4842215684522644, places=7)

    def test_seasonality_qs(self) -> None:
        res_qs = seasonality(self.data_airline, algorithm="qs", freq=12)
        assert isinstance(res_qs, tuple)
        assert_almost_equal(res_qs[0], self.result_qs_simple[0], places=5)

    def test_seasonality_ocsb(self) -> None:
        res_ocsb = seasonality(self.data_airline, algorithm="ocsb", m=12)
        assert res_ocsb == self.result_ocsb

    def test_seasonality_ch(self) -> None:
        res_ch = seasonality(self.data_airline, algorithm="ch", m=12)
        assert res_ch == self.result_ch

    def test_seasonality_ss(self) -> None:
        res_ss = seasonality(self.data_airline, algorithm="seasonal_strength", m=12)
        assert res_ss == self.result_seasonal_strength

    def test_seasonality_ts(self) -> None:
        res_ts = seasonality(self.data_airline, algorithm="trend_strength", m=12)
        assert res_ts == self.result_trend_strength

    def test_seasonality_s(self) -> None:
        res_sp = seasonality(self.data_airline, algorithm="spikiness", m=12)
        assert res_sp == self.result_spikiness

    def test_seasonality_error(self) -> None:
        with raises(ValueError):
            seasonality(self.data_airline, algorithm="invalid")

    def test_is_seasonal_qs(self) -> None:
        res_qs = is_seasonal(self.data_airline, algorithm="qs", freq=12)
        assert res_qs["result"] is True
        assert res_qs["algorithm"] == "qs"

    def test_is_seasonal_ocsb(self) -> None:
        res_ocsb = is_seasonal(self.data_airline, algorithm="ocsb", m=12)
        assert res_ocsb["result"] is True

    def test_is_seasonal_ch(self) -> None:
        res_ch = is_seasonal(self.data_airline, algorithm="ch", m=12)
        assert res_ch["result"] is False

    def test_is_seasonal_ss(self) -> None:
        res_ss = is_seasonal(self.data_airline, algorithm="seasonal_strength", m=12)
        assert res_ss["result"] is True  # 0.77 > 0.64

    def test_is_seasonal_ts(self) -> None:
        res_sp = is_seasonal(self.data_airline, algorithm="spikiness", m=12)
        assert res_sp["result"] is True
