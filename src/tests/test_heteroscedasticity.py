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
from statsmodels.stats.diagnostic import ResultsStore

# ## Local First Party Imports ----
from ts_stat_tests.algorithms.heteroscedasticity import arch, bpl, gq, wlm
from ts_stat_tests.tests.heteroscedasticity import (
    heteroscedasticity,
    is_heteroscedastic,
)


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Test Class                                                             ####
#                                                                              #
# ---------------------------------------------------------------------------- #


class TestHeteroscedasticity(unittest.TestCase):
    """Unit tests for heteroscedasticity algorithms and dispatcher."""

    def setUp(self) -> None:
        """Set up test data."""
        self.rng = np.random.default_rng(42)
        self.n_obs = 100
        self.x = sm.add_constant(np.linspace(1, 10, self.n_obs))

        # Homoscedastic data
        self.y_homo = 2 * self.x[:, 1] + self.rng.normal(scale=1.0, size=self.n_obs)
        self.res_homo = sm.OLS(self.y_homo, self.x).fit()

        # Heteroscedastic data (variance increases with x)
        self.y_het = 2 * self.x[:, 1] + self.rng.normal(scale=0.5 * self.x[:, 1], size=self.n_obs)
        self.res_het = sm.OLS(self.y_het, self.x).fit()

        # ARCH data (volatility clustering)
        # e_t = sigma_t * epsilon_t, sigma_t^2 = alpha_0 + alpha_1 * e_{t-1}^2
        # Use a simple ARCH(1) process
        arch_errors = np.zeros(self.n_obs)
        sigma2 = np.zeros(self.n_obs)
        epsilon = self.rng.normal(size=self.n_obs)
        sigma2[0] = 1.0
        arch_errors[0] = np.sqrt(sigma2[0]) * epsilon[0]
        for t in range(1, self.n_obs):
            sigma2[t] = 0.5 + 0.8 * arch_errors[t - 1] ** 2
            arch_errors[t] = np.sqrt(sigma2[t]) * epsilon[t]
        self.y_arch = 2 * self.x[:, 1] + arch_errors
        self.res_arch = sm.OLS(self.y_arch, self.x).fit()

    # ## Algorithm Tests ----

    def test_arch_results(self) -> None:
        """Test ARCH test results."""
        # store=False (default)
        res = arch(self.res_arch.resid)
        assert len(res) == 4
        for val in res:
            assert isinstance(val, float)

        # store=True
        res_store = arch(self.res_arch.resid, store=True)
        assert len(res_store) == 5
        assert isinstance(res_store[-1], ResultsStore)

    def test_bpl_results(self) -> None:
        """Test Breusch-Pagan test results."""
        res = bpl(self.res_het.resid, self.x)
        assert len(res) == 4
        for val in res:
            assert isinstance(val, float)

    def test_gq_results(self) -> None:
        """Test Goldfeld-Quandt test results."""
        # store=False (default)
        res = gq(self.y_het, self.x)
        assert len(res) == 3
        assert isinstance(res[0], float)
        assert isinstance(res[1], float)
        assert isinstance(res[2], str)

        # store=True
        res_store = gq(self.y_het, self.x, store=True)
        assert len(res_store) == 4
        assert isinstance(res_store[-1], ResultsStore)

    def test_wlm_results(self) -> None:
        """Test White's test results."""
        res = wlm(self.res_het.resid, self.x)
        assert len(res) == 4
        for val in res:
            assert isinstance(val, float)

    # ## Dispatcher Tests ----

    def test_heteroscedasticity_dispatcher(self) -> None:
        """Test the heteroscedasticity dispatcher."""
        # Test all algorithms through dispatcher
        for alg in ["arch", "bp", "gq", "white"]:
            res = heteroscedasticity(self.res_homo, algorithm=alg)
            if alg == "gq":
                assert len(res) == 3
            else:
                assert len(res) == 4

        # Test invalid algorithm
        with raises(ValueError):
            heteroscedasticity(self.res_homo, algorithm="invalid")

    def test_is_heteroscedastic_results(self) -> None:
        """Test is_heteroscedastic function."""
        # Check homoscedastic case
        res_check = is_heteroscedastic(self.res_homo, algorithm="bp")
        assert isinstance(res_check["result"], bool)
        assert isinstance(res_check["pvalue"], float)
        assert not res_check["result"]  # Expect False for homoscedastic

        # Check heteroscedastic case
        res_check_het = is_heteroscedastic(self.res_het, algorithm="bp")
        assert res_check_het["result"]  # Expect True for heteroscedastic

    def test_is_heteroscedastic_with_store(self) -> None:
        """Test is_heteroscedastic with store=True to cover ResultsStore path."""
        # ARCH with store=True
        res_arch_store = is_heteroscedastic(self.res_arch, algorithm="arch", store=True)
        assert isinstance(res_arch_store["result"], bool)
        assert "statistic" in res_arch_store

        # GQ with store=True
        res_gq_store = is_heteroscedastic(self.res_het, algorithm="gq", store=True)
        assert isinstance(res_gq_store["result"], bool)

    def test_is_heteroscedastic_all_algs(self) -> None:
        """Test is_heteroscedastic with all algorithms."""
        for alg in ["arch", "bp", "gq", "white"]:
            res = is_heteroscedastic(self.res_homo, algorithm=alg)
            assert res["algorithm"] == alg
            assert "result" in res
            assert "pvalue" in res
