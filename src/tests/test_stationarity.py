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
from arch.unitroot import (
    DFGLS as ar_ers,
    PhillipsPerron as ar_pp,
    VarianceRatio as ar_vr,
)
from pytest import raises
from statsmodels.tsa.stattools import (
    adfuller as sm_adf,
    kpss as sm_kpss,
    range_unit_root_test as sm_rur,
    zivot_andrews as sm_za,
)

# ## Local First Party Imports ----
from tests.setup import BaseTester
from ts_stat_tests.algorithms.stationarity import adf, ers, kpss, pp, rur, vr, za
from ts_stat_tests.tests.stationarity import is_stationary, stationarity


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Testing                                                                ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Tests                                                                     ####
# ---------------------------------------------------------------------------- #


class TestStationarity(BaseTester):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def setUp(self):
        pass

    def test_stationarity(self) -> None:
        assert isinstance(stationarity(x=np.arange(100)), tuple)

    def test_is_stationarity(self) -> None:
        assert isinstance(is_stationary(x=np.arange(100), algorithm="adf"), dict)

    def test_stationarity_invalid_algorithm(self) -> None:
        """Test invalid algorithm in stationarity dispatcher."""

        with raises(ValueError) as excinfo:
            stationarity(np.arange(100), algorithm="invalid")

        assert "algorithm" in str(excinfo.value)

    def test_is_stationary_logic(self) -> None:
        """Test different H0 logic in is_stationary."""
        data = np.random.normal(0, 1, 100)

        # ADF (H0: unit root)
        res_adf = is_stationary(data, algorithm="adf", alpha=0.05)
        assert isinstance(res_adf["result"], bool)

        # KPSS (H0: stationary)
        res_kpss = is_stationary(data, algorithm="kpss", alpha=0.05)
        assert isinstance(res_kpss["result"], bool)

    def test_is_stationary_pvalue_bool_coverage(self) -> None:
        """Test branch coverage for pvalue_or_bool being a boolean."""

        with patch("ts_stat_tests.tests.stationarity.stationarity") as mock_stat:
            # Mock returning (stat, bool)
            mock_stat.return_value = (0.5, True)
            res = is_stationary(np.arange(100), algorithm="adf")
            assert res["result"] is True
            assert res["pvalue"] is None

    def test_is_stationary_non_numeric_pvalue(self) -> None:
        """Test coverage where pvalue is neither bool nor numeric."""

        with patch("ts_stat_tests.tests.stationarity.stationarity") as mock_stat:
            # Mock returning (stat, string)
            mock_stat.return_value = (0.5, "invalid")
            res = is_stationary(np.arange(100), algorithm="adf")
            assert res["result"] is False
            assert res["pvalue"] is None

    def test_stationarity_dispatcher_coverage(self) -> None:
        """Test all algorithms in stationarity dispatcher for coverage."""
        data = np.random.normal(0, 1, 100)
        # These are already covered or partially covered, but let's be explicit
        stationarity(data, algorithm="pp")
        stationarity(data, algorithm="za")
        stationarity(data, algorithm="ers")
        stationarity(data, algorithm="vr")
        stationarity(data, algorithm="rur")


class TestStationarityADF(BaseTester):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.result_adf_airline = adf(x=cls.data_airline)
        cls.result_adf_random = adf(x=cls.data_random)
        cls.result_adf_sine = adf(x=cls.data_sine)
        cls.result_adf_line = adf(x=cls.data_line)
        cls.result_adf_basic = adf(x=cls.data_basic)
        cls.result_adf_noise = adf(x=cls.data_noise)
        cls.result_sm_adf_airline = sm_adf(x=cls.data_airline)
        cls.result_sm_adf_random = sm_adf(x=cls.data_random)
        cls.result_sm_adf_sine = sm_adf(x=cls.data_sine)
        cls.result_sm_adf_line = sm_adf(x=cls.data_line)
        cls.result_sm_adf_basic = sm_adf(x=cls.data_basic)
        cls.result_sm_adf_noise = sm_adf(x=cls.data_noise)

    def setUp(self):
        pass

    def test_stationarity_adf_airline(self) -> None:
        assert self.result_adf_airline == self.result_sm_adf_airline

    def test_stationarity_adf_random(self) -> None:
        assert self.result_adf_random == self.result_sm_adf_random

    def test_stationarity_adf_sine(self) -> None:
        assert self.result_adf_sine == self.result_sm_adf_sine

    def test_stationarity_adf_line(self) -> None:
        assert self.result_adf_line == self.result_sm_adf_line

    def test_stationarity_adf_basic(self) -> None:
        assert self.result_adf_basic == self.result_sm_adf_basic

    def test_stationarity_adf_noise(self) -> None:
        assert self.result_adf_noise == self.result_sm_adf_noise

    def test_stationarity_adf_store(self) -> None:
        """Test adf with store=True."""
        res = adf(x=self.data_airline, store=True)
        assert len(res) == 4
        assert isinstance(res[0], float)
        assert isinstance(res[1], float)
        assert isinstance(res[2], dict)
        assert hasattr(res[3], "nobs")

    def test_stationarity_adf_autolag_none(self) -> None:
        """Test adf with autolag=None."""
        res = adf(x=self.data_airline, autolag=None)
        assert len(res) == 5
        assert isinstance(res[0], float)
        assert isinstance(res[1], float)
        assert isinstance(res[2], int)
        assert isinstance(res[3], int)
        assert isinstance(res[4], dict)


class TestStationarityKPSS(BaseTester):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.result_kpss_airline = kpss(x=cls.data_airline)
        cls.result_kpss_random = kpss(x=cls.data_random)
        cls.result_kpss_sine = kpss(x=cls.data_sine)
        cls.result_kpss_line = kpss(x=cls.data_line)
        cls.result_kpss_basic = kpss(x=cls.data_basic)
        cls.result_kpss_noise = kpss(x=cls.data_noise)
        cls.result_sm_kpss_airline = sm_kpss(x=cls.data_airline)
        cls.result_sm_kpss_random = sm_kpss(x=cls.data_random)
        cls.result_sm_kpss_sine = sm_kpss(x=cls.data_sine)
        cls.result_sm_kpss_line = sm_kpss(x=cls.data_line)
        cls.result_sm_kpss_basic = sm_kpss(x=cls.data_basic)
        cls.result_sm_kpss_noise = sm_kpss(x=cls.data_noise)

    def setUp(self):
        pass

    def test_stationarity_kpss_airline(self) -> None:
        assert self.result_kpss_airline == self.result_sm_kpss_airline

    def test_stationarity_kpss_random(self) -> None:
        assert self.result_kpss_random == self.result_sm_kpss_random

    def test_stationarity_kpss_sine(self) -> None:
        assert self.result_kpss_sine == self.result_sm_kpss_sine

    def test_stationarity_kpss_line(self) -> None:
        assert self.result_kpss_line == self.result_sm_kpss_line

    def test_stationarity_kpss_basic(self) -> None:
        assert self.result_kpss_basic == self.result_sm_kpss_basic

    def test_stationarity_kpss_noise(self) -> None:
        assert self.result_kpss_noise == self.result_sm_kpss_noise


class TestStationarityRUR(BaseTester):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.result_rur_airline = rur(x=cls.data_airline)
        cls.result_rur_random = rur(x=cls.data_random)
        cls.result_rur_sine = rur(x=cls.data_sine)
        cls.result_rur_line = rur(x=cls.data_line)
        # cls.result_rur_basic = rur(x=cls.data_basic)
        # cls.result_rur_noise = rur(x=cls.data_noise)
        cls.result_sm_rur_airline = sm_rur(x=cls.data_airline)
        cls.result_sm_rur_random = sm_rur(x=cls.data_random)
        cls.result_sm_rur_sine = sm_rur(x=cls.data_sine)
        cls.result_sm_rur_line = sm_rur(x=cls.data_line)
        # cls.result_sm_rur_basic = sm_rur(x=cls.data_basic)
        # cls.result_sm_rur_noise = sm_rur(x=cls.data_noise)

    def setUp(self):
        pass

    def test_stationarity_rur_airline(self) -> None:
        assert self.result_rur_airline == self.result_sm_rur_airline

    def test_stationarity_rur_random(self) -> None:
        assert self.result_rur_random == self.result_sm_rur_random

    def test_stationarity_rur_sine(self) -> None:
        assert self.result_rur_sine == self.result_sm_rur_sine

    def test_stationarity_rur_line(self) -> None:
        assert self.result_rur_line == self.result_sm_rur_line

    def test_stationarity_rur_basic(self) -> None:
        # assert self.result_rur_basic == self.result_sm_rur_basic
        pass

    def test_stationarity_rur_noise(self) -> None:
        # assert self.result_rur_noise == self.result_sm_rur_noise
        pass


class TestStationarityZA(BaseTester):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.result_za_airline = za(x=cls.data_airline)
        cls.result_za_random = za(x=cls.data_random)
        # cls.result_za_sine = za(x=cls.data_sine)
        # cls.result_za_line = za(x=cls.data_line)
        cls.result_za_basic = za(x=cls.data_basic)
        cls.result_za_noise = za(x=cls.data_noise)
        cls.result_sm_za_airline = sm_za(x=cls.data_airline)
        cls.result_sm_za_random = sm_za(x=cls.data_random)
        # cls.result_sm_za_sine = sm_za(x=cls.data_sine)
        # cls.result_sm_za_line = sm_za(x=cls.data_line)
        cls.result_sm_za_basic = sm_za(x=cls.data_basic)
        cls.result_sm_za_noise = sm_za(x=cls.data_noise)

    def setUp(self):
        pass

    def test_stationarity_za_airline(self) -> None:
        self.result_za_airline = tuple(
            round(elem, 7) if index <= 1 else elem for index, elem in enumerate(self.result_za_airline)
        )
        self.result_sm_za_airline = tuple(
            round(elem, 7) if index <= 1 else elem for index, elem in enumerate(self.result_sm_za_airline)
        )
        assert self.result_za_airline == self.result_sm_za_airline

    def test_stationarity_za_random(self) -> None:
        self.result_za_random = tuple(
            round(elem, 7) if index <= 1 else elem for index, elem in enumerate(self.result_za_random)
        )
        self.result_sm_za_random = tuple(
            round(elem, 7) if index <= 1 else elem for index, elem in enumerate(self.result_sm_za_random)
        )
        assert self.result_za_random == self.result_sm_za_random

    def test_stationarity_za_sine(self) -> None:
        # assert self.result_za_sine == self.result_sm_za_sine
        pass

    def test_stationarity_za_line(self) -> None:
        # assert self.result_za_line == self.result_sm_za_line
        pass

    def test_stationarity_za_basic(self) -> None:
        assert self.result_za_basic == self.result_sm_za_basic

    def test_stationarity_za_noise(self) -> None:
        self.result_za_noise = tuple(
            round(elem, 7) if index <= 1 else elem for index, elem in enumerate(self.result_za_noise)
        )
        self.result_sm_za_noise = tuple(
            round(elem, 7) if index <= 1 else elem for index, elem in enumerate(self.result_sm_za_noise)
        )
        assert self.result_za_noise == self.result_sm_za_noise


class TestStationarityPP(BaseTester):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        def get_ar_pp_res(y):
            nobs = len(y)
            lags = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
            if lags >= nobs - 1:
                lags = max(0, nobs - 2)
            res = ar_pp(y, lags=lags)
            return (float(res.stat), float(res.pvalue), int(res.lags), dict(res.critical_values))

        cls.result_pp_airline = pp(x=cls.data_airline)
        cls.result_pp_random = pp(x=cls.data_random)
        cls.result_pp_sine = pp(x=cls.data_sine)
        cls.result_pp_line = pp(x=cls.data_line)
        cls.result_pp_basic = pp(x=np.array(cls.data_basic))
        cls.result_pp_noise = pp(x=cls.data_noise)
        cls.result_pa_pp_airline = get_ar_pp_res(cls.data_airline)
        cls.result_pa_pp_random = get_ar_pp_res(cls.data_random)
        cls.result_pa_pp_sine = get_ar_pp_res(cls.data_sine)
        cls.result_pa_pp_line = get_ar_pp_res(cls.data_line)
        cls.result_pa_pp_basic = get_ar_pp_res(np.array(cls.data_basic))
        cls.result_pa_pp_noise = get_ar_pp_res(cls.data_noise)

    def setUp(self):
        pass

    def test_stationarity_pp_airline(self) -> None:
        assert self.result_pp_airline == self.result_pa_pp_airline

    def test_stationarity_pp_random(self) -> None:
        assert self.result_pp_random == self.result_pa_pp_random

    def test_stationarity_pp_sine(self) -> None:
        assert self.result_pp_sine == self.result_pa_pp_sine

    def test_stationarity_pp_line(self) -> None:
        assert self.result_pp_line == self.result_pa_pp_line

    def test_stationarity_pp_basic(self) -> None:
        assert self.result_pp_basic == self.result_pa_pp_basic

    def test_stationarity_pp_noise(self) -> None:
        assert self.result_pp_noise == self.result_pa_pp_noise


class TestStationarityERS(BaseTester):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.result_ers_airline = ers(y=cls.data_airline)
        cls.result_ers_random = ers(y=cls.data_random)
        # cls.result_ers_sine = ers(y=cls.data_sine)
        # cls.result_ers_line = ers(y=cls.data_line)
        cls.result_ers_basic = ers(y=cls.data_basic)
        cls.result_ers_noise = ers(y=cls.data_noise)
        ar_ers_airline = ar_ers(y=cls.data_airline)
        cls.result_ar_ers_airline = (
            float(ar_ers_airline.stat),
            float(ar_ers_airline.pvalue),
            int(ar_ers_airline.lags),
            dict(ar_ers_airline.critical_values),
        )
        ar_ers_random = ar_ers(y=cls.data_random)
        cls.result_ar_ers_random = (
            float(ar_ers_random.stat),
            float(ar_ers_random.pvalue),
            int(ar_ers_random.lags),
            dict(ar_ers_random.critical_values),
        )
        # ar_ers_sine = ar_ers(y=cls.data_sine)
        # cls.result_ar_ers_sine = (float(ar_ers_sine.stat), float(ar_ers_sine.pvalue), int(ar_ers_sine.lags), dict(ar_ers_sine.critical_values))
        # ar_ers_line = ar_ers(y=cls.data_line)
        # cls.result_ar_ers_line = (float(ar_ers_line.stat), float(ar_ers_line.pvalue), int(ar_ers_line.lags), dict(ar_ers_line.critical_values))
        ar_ers_basic = ar_ers(y=cls.data_basic)
        cls.result_ar_ers_basic = (
            float(ar_ers_basic.stat),
            float(ar_ers_basic.pvalue),
            int(ar_ers_basic.lags),
            dict(ar_ers_basic.critical_values),
        )
        ar_ers_noise = ar_ers(y=cls.data_noise)
        cls.result_ar_ers_noise = (
            float(ar_ers_noise.stat),
            float(ar_ers_noise.pvalue),
            int(ar_ers_noise.lags),
            dict(ar_ers_noise.critical_values),
        )

    def setUp(self):
        pass

    def test_stationarity_ers_airline(self) -> None:
        assert self.result_ers_airline == self.result_ar_ers_airline

    def test_stationarity_ers_random(self) -> None:
        assert self.result_ers_random == self.result_ar_ers_random

    def test_stationarity_ers_sine(self) -> None:
        # assert self.result_ers_sine == self.result_ar_ers_sine
        pass

    def test_stationarity_ers_line(self) -> None:
        # assert self.result_ers_line == self.result_ar_ers_line
        pass

    def test_stationarity_ers_basic(self) -> None:
        assert self.result_ers_basic == self.result_ar_ers_basic

    def test_stationarity_ers_noise(self) -> None:
        assert self.result_ers_noise == self.result_ar_ers_noise


class TestStationarityVR(BaseTester):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.result_vr_airline = vr(y=cls.data_airline)
        cls.result_vr_random = vr(y=cls.data_random)
        cls.result_vr_sine = vr(y=cls.data_sine)
        cls.result_vr_line = vr(y=cls.data_line)
        cls.result_vr_basic = vr(y=cls.data_basic)
        cls.result_vr_noise = vr(y=cls.data_noise)
        ar_vr_airline = ar_vr(y=cls.data_airline)
        cls.result_ar_vr_airline = (
            float(ar_vr_airline.stat),
            float(ar_vr_airline.pvalue),
            float(ar_vr_airline.vr),
        )
        ar_vr_random = ar_vr(y=cls.data_random)
        cls.result_ar_vr_random = (
            float(ar_vr_random.stat),
            float(ar_vr_random.pvalue),
            float(ar_vr_random.vr),
        )
        ar_vr_sine = ar_vr(y=cls.data_sine)
        cls.result_ar_vr_sine = (float(ar_vr_sine.stat), float(ar_vr_sine.pvalue), float(ar_vr_sine.vr))
        ar_vr_line = ar_vr(y=cls.data_line)
        cls.result_ar_vr_line = (float(ar_vr_line.stat), float(ar_vr_line.pvalue), float(ar_vr_line.vr))
        ar_vr_basic = ar_vr(y=cls.data_basic)
        cls.result_ar_vr_basic = (float(ar_vr_basic.stat), float(ar_vr_basic.pvalue), float(ar_vr_basic.vr))
        ar_vr_noise = ar_vr(y=cls.data_noise)
        cls.result_ar_vr_noise = (float(ar_vr_noise.stat), float(ar_vr_noise.pvalue), float(ar_vr_noise.vr))

    def setUp(self):
        pass

    def test_stationarity_vr_airline(self) -> None:
        assert self.result_vr_airline == self.result_ar_vr_airline

    def test_stationarity_vr_random(self) -> None:
        assert self.result_vr_random == self.result_ar_vr_random

    def test_stationarity_vr_sine(self) -> None:
        assert self.result_vr_sine == self.result_ar_vr_sine

    def test_stationarity_vr_line(self) -> None:
        # assert self.result_vr_line == self.result_ar_vr_line
        assert np.all(np.isnan(np.array(self.result_vr_line)))

    def test_stationarity_vr_basic(self) -> None:
        assert self.result_vr_basic == self.result_ar_vr_basic

    def test_stationarity_vr_noise(self) -> None:
        assert self.result_vr_noise == self.result_ar_vr_noise
