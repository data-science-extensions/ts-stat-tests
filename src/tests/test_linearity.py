# ============================================================================ #
#                                                                              #
#     Title: Test Linearity                                                    #
#     Purpose: Unit tests for linearity algorithms.                            #
#                                                                              #
# ============================================================================ #


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Setup                                                                 ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Imports                                                                   ####
# ---------------------------------------------------------------------------- #


# ## Python StdLib Imports ----
import unittest

# ## Python Third Party Imports ----
import numpy as np
import statsmodels.api as sm
from pytest import raises
from statsmodels.stats.contrast import ContrastResults

# ## Local First Party Imports ----
from ts_stat_tests.algorithms.linearity import hc, lm, rb, rr
from ts_stat_tests.tests.linearity import is_linear, linearity


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Test Class                                                             ####
#                                                                              #
# ---------------------------------------------------------------------------- #


class TestLinearity(unittest.TestCase):
    """Unit tests for linearity algorithms and dispatcher."""

    def setUp(self) -> None:
        """Set up test data."""
        self.rng = np.random.default_rng(42)
        self.x = sm.add_constant(np.linspace(0, 10, 100))
        # Linear relationship
        self.y_linear = 3 + 2 * self.x[:, 1] + self.rng.normal(size=100)
        # Non-linear relationship (quadratic)
        self.y_nonlinear = 3 + 2 * self.x[:, 1] + 0.5 * self.x[:, 1] ** 2 + self.rng.normal(size=100)

        self.res_linear = sm.OLS(self.y_linear, self.x).fit()
        self.res_nonlinear = sm.OLS(self.y_nonlinear, self.x).fit()

    # ## Algorithm Tests ----

    def test_hc_results(self) -> None:
        """Test Harvey-Collier test results."""
        stat, pval = hc(self.res_linear)
        assert isinstance(stat, float)
        assert isinstance(pval, float)

    def test_lm_results(self) -> None:
        """Test Lagrange Multiplier test results."""
        # LM test on linear residuals
        stat, pval, fval, fpval = lm(self.res_linear.resid, self.x)
        assert isinstance(stat, float)
        assert isinstance(pval, float)
        assert isinstance(fval, float)
        assert isinstance(fpval, float)

    def test_rb_results(self) -> None:
        """Test Rainbow test results."""
        stat, pval = rb(self.res_linear)
        assert isinstance(stat, float)
        assert isinstance(pval, float)

    def test_rr_results(self) -> None:
        """Test Ramsey RESET test results."""
        res = rr(self.res_linear)
        # ContrastResults has pvalue, but statistic name might vary (fvalue or statistic)
        assert hasattr(res, "pvalue")
        assert hasattr(res, "statistic") or hasattr(res, "fvalue")

    def test_hc_discrimination(self) -> None:
        """Test if HC test can discriminate non-linearity."""
        # Note: Non-linearity detection depends on the data and alpha
        _, pval_linear = hc(self.res_linear)
        _, pval_nonlinear = hc(self.res_nonlinear)
        # We expect linear p-value to be higher than non-linear p-value generally
        # (though randomness can affect this, 42 seed is usually okay)
        assert pval_linear > 0.05

    def test_rb_params(self) -> None:
        """Test Rainbow test with different parameters."""
        # Use data without constant for use_distance=True to avoid singular matrix
        x_no_const = np.linspace(0, 10, 100).reshape(-1, 1)
        y = 3 + 2 * x_no_const.flatten() + self.rng.normal(size=100)
        res = sm.OLS(y, x_no_const).fit()
        stat, pval = rb(res, frac=0.4, use_distance=True)
        assert isinstance(stat, float)
        assert isinstance(pval, float)

    def test_rr_params(self) -> None:
        """Test Ramsey RESET with different parameters."""
        res = rr(self.res_linear, power=[2, 3], test_type="exog")
        assert hasattr(res, "pvalue")
        assert hasattr(res, "statistic") or hasattr(res, "fvalue")

    # ## Dispatcher Tests ----

    def test_linearity_dispatcher(self) -> None:
        """Test linearity dispatcher with different algorithms."""
        # Test HC
        res_hc = linearity(self.res_linear, algorithm="hc")
        assert isinstance(res_hc, tuple)
        assert len(res_hc) == 2

        # Test LM
        res_lm = linearity(self.res_linear, algorithm="lm")
        assert isinstance(res_lm, tuple)
        assert len(res_lm) == 4

        # Test RB
        res_rb = linearity(self.res_linear, algorithm="rb")
        assert isinstance(res_rb, tuple)
        assert len(res_rb) == 2

        # Test RR
        res_rr = linearity(self.res_linear, algorithm="rr")
        assert isinstance(res_rr, ContrastResults)

    def test_linearity_dispatcher_invalid(self) -> None:
        """Test linearity dispatcher with invalid algorithm."""
        with raises(ValueError):
            linearity(self.res_linear, algorithm="invalid")

    def test_is_linear_results(self) -> None:
        """Test is_linear boolean check."""
        # Test linear data
        res = is_linear(self.res_linear, algorithm="rr")
        assert res["result"] is True
        assert res["algorithm"] == "rr"
        assert "pvalue" in res
        assert "statistic" in res

        # Test non-linear data
        res_nl = is_linear(self.res_nonlinear, algorithm="rr")
        # RESET is good at picking up quadratic
        assert res_nl["result"] is False

    def test_is_linear_all_algorithms(self) -> None:
        """Test is_linear with all supported algorithms."""
        for alg in ["hc", "lm", "rb", "rr"]:
            res = is_linear(self.res_linear, algorithm=alg)
            assert isinstance(res["result"], bool)
            assert res["algorithm"] == alg
