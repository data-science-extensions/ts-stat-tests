# ---------------------------------------------------------------------------- #
#                                                                              #
#    Setup                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Imports                                                                   ####
# ---------------------------------------------------------------------------- #


# ## Python Third Party Imports ----
from tsfeatures import lumpiness as ts_lumpiness, stability as ts_stability

# ## Local First Party Imports ----
from tests.setup import BaseTester
from ts_stat_tests.algorithms.stability import (
    lumpiness,
    stability,
)
from ts_stat_tests.tests.stability import (
    is_lumpy,
    is_stable,
)


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Testing                                                                ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Tests                                                                     ####
# ---------------------------------------------------------------------------- #


class TestStability(BaseTester):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.result_stability = stability(cls.data_airline)
        cls.result_is_stable = is_stable(cls.data_airline)
        cls.result_lumpiness = lumpiness(cls.data_airline)
        cls.result_is_lumpy = is_lumpy(cls.data_airline)

    def setUp(self) -> None:
        pass

    def test_stability_1(self) -> None:
        result = self.result_stability
        expected = ts_stability(self.data_airline)["stability"]
        assert result == expected

    def test_stability_2(self) -> None:
        result = self.result_is_stable
        expected = False
        assert result == expected

    def test_lumpiness_1(self) -> None:
        result = self.result_lumpiness
        expected = ts_lumpiness(self.data_airline)["lumpiness"]
        assert result == expected

    def test_lumpiness_2(self) -> None:
        result = self.result_is_lumpy
        expected = True
        assert result == expected
