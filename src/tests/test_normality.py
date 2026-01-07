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

# ## Local First Party Imports ----
from tests.setup import BaseTester
from ts_stat_tests.algorithms.normality import (
    ad,
    dp,
    jb,
    ob,
    sw,
)
from ts_stat_tests.tests.normality import is_normal, normality
from ts_stat_tests.utils.errors import assert_almost_equal


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Testing                                                                ####
#                                                                              #
# ---------------------------------------------------------------------------- #


class TestNormality(BaseTester):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Seed for reproducibility
        rng = np.random.default_rng(42)
        cls.data_normal = rng.normal(0, 1, 100)
        cls.data_non_normal = rng.exponential(1, 100)

        cls.result_jb = is_normal(x=cls.data_normal, algorithm="jb")
        cls.result_ob = is_normal(x=cls.data_normal, algorithm="ob")
        cls.result_sw = is_normal(x=cls.data_normal, algorithm="sw")
        cls.result_dp = is_normal(x=cls.data_normal, algorithm="dp")
        cls.result_ad = is_normal(x=cls.data_normal, algorithm="ad")

    def test_normality_keys(self) -> None:
        assert set(self.result_dp.keys()) >= {"result", "statistic", "p_value", "alpha"}

    def test_normality_ad_keys(self) -> None:
        assert set(self.result_ad.keys()) >= {"result", "statistic", "critical_value", "significance_level", "alpha"}

    def test_normality_types(self) -> None:
        assert isinstance(self.result_dp["result"], bool)
        assert isinstance(self.result_dp["statistic"], (float, np.floating))
        assert isinstance(self.result_dp["p_value"], (float, np.floating))

    def test_is_normal_results(self) -> None:
        # With enough samples, normal data should be normal
        assert self.result_sw["result"] is True

        # Exponential data should not be normal
        res_non_normal = is_normal(x=self.data_non_normal, algorithm="sw")
        assert res_non_normal["result"] is False

    def test_normality_dispatch(self) -> None:
        # Check that dispatch works for different names
        res1 = normality(self.data_normal, algorithm="jarque-bera")
        res2 = jb(self.data_normal)
        assert_almost_equal(res1[0], res2[0])

    def test_normality_errors(self) -> None:
        with raises(ValueError):
            normality(self.data_normal, algorithm="invalid")

    def test_jb_basic(self) -> None:
        res = jb(self.data_normal)
        assert len(res) == 4  # stat, pval, skew, kurtosis

    def test_ob_basic(self) -> None:
        res = ob(self.data_normal)
        assert len(res) == 2

    def test_sw_basic(self) -> None:
        res = sw(self.data_normal)
        assert len(res) == 2

    def test_dp_basic(self) -> None:
        res = dp(self.data_normal)
        assert len(res) == 2

    def test_ad_basic(self) -> None:
        res = ad(self.data_normal)
        assert len(res) == 3
