# ---------------------------------------------------------------------------- #
#                                                                              #
#    Setup                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Imports                                                                   ####
# ---------------------------------------------------------------------------- #


# ## Python StdLib Imports ----

# ## Python Third Party Imports ----
import numpy as np
from arch.unitroot import DFGLS as ar_ers, VarianceRatio as ar_vr
from pmdarima.arima import PPTest as pa_pp
from statsmodels.tsa.stattools import (
    adfuller as sm_adf,
    kpss as sm_kpss,
    range_unit_root_test as sm_rur,
    zivot_andrews as sm_za,
)

# ## Local First Party Imports ----
from src.tests.test_base import BaseTester
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
        self.assertIsNone(stationarity(x=np.arange(100)))

    def test_is_stationarity(self) -> None:
        self.assertIsNone(is_stationary(x=np.arange(100)))


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

    def test_statuionarity_za_airline(self) -> None:
        self.result_za_airline = tuple(
            round(elem, 7) if indx <= 1 else elem for indx, elem in enumerate(self.result_za_airline)
        )
        self.result_sm_za_airline = tuple(
            round(elem, 7) if indx <= 1 else elem for indx, elem in enumerate(self.result_sm_za_airline)
        )
        assert self.result_za_airline == self.result_sm_za_airline

    def test_statuionarity_za_random(self) -> None:
        self.result_za_random = tuple(
            round(elem, 7) if indx <= 1 else elem for indx, elem in enumerate(self.result_za_random)
        )
        self.result_sm_za_random = tuple(
            round(elem, 7) if indx <= 1 else elem for indx, elem in enumerate(self.result_sm_za_random)
        )
        assert self.result_za_random == self.result_sm_za_random

    def test_statuionarity_za_sine(self) -> None:
        # assert self.result_za_sine == self.result_sm_za_sine
        pass

    def test_statuionarity_za_line(self) -> None:
        # assert self.result_za_line == self.result_sm_za_line
        pass

    def test_statuionarity_za_basic(self) -> None:
        assert self.result_za_basic == self.result_sm_za_basic

    def test_statuionarity_za_noise(self) -> None:
        self.result_za_noise = tuple(
            round(elem, 7) if indx <= 1 else elem for indx, elem in enumerate(self.result_za_noise)
        )
        self.result_sm_za_noise = tuple(
            round(elem, 7) if indx <= 1 else elem for indx, elem in enumerate(self.result_sm_za_noise)
        )
        assert self.result_za_noise == self.result_sm_za_noise


class TestStationarityPP(BaseTester):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.result_pp_airline = pp(x=cls.data_airline)
        cls.result_pp_random = pp(x=cls.data_random)
        cls.result_pp_sine = pp(x=cls.data_sine)
        cls.result_pp_line = pp(x=cls.data_line)
        cls.result_pp_basic = pp(x=np.array(cls.data_basic))
        cls.result_pp_noise = pp(x=cls.data_noise)
        cls.result_pa_pp_airline = pa_pp().should_diff(x=cls.data_airline)
        cls.result_pa_pp_random = pa_pp().should_diff(x=cls.data_random)
        cls.result_pa_pp_sine = pa_pp().should_diff(x=cls.data_sine)
        cls.result_pa_pp_line = pa_pp().should_diff(x=cls.data_line)
        cls.result_pa_pp_basic = pa_pp().should_diff(x=np.array(cls.data_basic))
        cls.result_pa_pp_noise = pa_pp().should_diff(x=cls.data_noise)

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
        cls.result_ar_ers_airline = (ar_ers_airline.pvalue, ar_ers_airline.stat)
        ar_ers_random = ar_ers(y=cls.data_random)
        cls.result_ar_ers_random = (ar_ers_random.pvalue, ar_ers_random.stat)
        # ar_ers_sine = ar_ers(y=cls.data_sine)
        # cls.result_ar_ers_sine = (ar_ers_sine.pvalue, ar_ers_sine.stat)
        # ar_ers_line = ar_ers(y=cls.data_line)
        # cls.result_ar_ers_line = (ar_ers_line.pvalue, ar_ers_line.stat)
        ar_ers_basic = ar_ers(y=cls.data_basic)
        cls.result_ar_ers_basic = (ar_ers_basic.pvalue, ar_ers_basic.stat)
        ar_ers_noise = ar_ers(y=cls.data_noise)
        cls.result_ar_ers_noise = (ar_ers_noise.pvalue, ar_ers_noise.stat)

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
            ar_vr_airline.pvalue,
            ar_vr_airline.stat,
            ar_vr_airline.vr,
        )
        ar_vr_random = ar_vr(y=cls.data_random)
        cls.result_ar_vr_random = (
            ar_vr_random.pvalue,
            ar_vr_random.stat,
            ar_vr_random.vr,
        )
        ar_vr_sine = ar_vr(y=cls.data_sine)
        cls.result_ar_vr_sine = (ar_vr_sine.pvalue, ar_vr_sine.stat, ar_vr_sine.vr)
        ar_vr_line = ar_vr(y=cls.data_line)
        cls.result_ar_vr_line = (ar_vr_line.pvalue, ar_vr_line.stat, ar_vr_line.vr)
        ar_vr_basic = ar_vr(y=cls.data_basic)
        cls.result_ar_vr_basic = (ar_vr_basic.pvalue, ar_vr_basic.stat, ar_vr_basic.vr)
        ar_vr_noise = ar_vr(y=cls.data_noise)
        cls.result_ar_vr_noise = (ar_vr_noise.pvalue, ar_vr_noise.stat, ar_vr_noise.vr)

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
