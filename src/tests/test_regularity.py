# ---------------------------------------------------------------------------- #
#                                                                              #
#    Setup                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Imports                                                                   ####
# ---------------------------------------------------------------------------- #


# ## Python Third Party Imports ----
from pytest import raises

# ## Local First Party Imports ----
from tests.setup import BaseTester
from ts_stat_tests.regularity import (
    approx_entropy,
    entropy,
    is_regular,
    permutation_entropy,
    sample_entropy,
    spectral_entropy,
    svd_entropy,
)
from ts_stat_tests.utils.errors import assert_almost_equal


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Testing                                                                ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Tests                                                                     ####
# ---------------------------------------------------------------------------- #


class TestRegularity(BaseTester):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.result_sample = is_regular(x=cls.data_airline, algorithm="sample")
        cls.result_approx = is_regular(x=cls.data_airline, algorithm="approx", tolerance=20)
        cls.result_spectral = is_regular(x=cls.data_airline, algorithm="spectral", sf=1)

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_regularity_keys(self) -> None:
        assert list(self.result_sample.keys()) == ["result", "entropy", "tolerance"]

    def test_regularity_types(self) -> None:
        params = [
            (self.result_sample["result"], bool),
            (self.result_sample["entropy"], (float, int)),
            (self.result_sample["tolerance"], (float, int)),
        ]
        for value, expected in params:
            assert isinstance(value, expected)

    def test_regularity_values(self) -> None:
        params = [
            (self.result_sample["entropy"], 0.6177074729583698),
            (self.result_sample["tolerance"], 23.909808306554297),
            (self.result_approx["entropy"], 0.6451264780416452),
            (self.result_approx["tolerance"], 20),
            (self.result_spectral["entropy"], 0.4287365561752448),
            (self.result_spectral["tolerance"], 23.909808306554297),
        ]
        for value, expected in params:
            assert_almost_equal(first=value, second=expected)

    def test_regularity_results(self) -> None:
        params = [
            (self.result_sample["result"], True),
            (self.result_approx["result"], True),
            (self.result_spectral["result"], True),
        ]
        for value, expected in params:
            assert value == expected

    def test_regularity_errors(self) -> None:
        with raises(ValueError):
            is_regular(x=self.data_airline, algorithm="error")
        with raises(ValueError):
            is_regular(x=self.data_airline, tolerance="error")

    def test_approx_basic(self) -> None:
        assert_almost_equal(
            first=approx_entropy(x=self.data_airline),
            second=0.6451,
            places=4,
        )

    def test_approx_noise(self) -> None:
        assert_almost_equal(
            first=approx_entropy(x=self.data_noise, order=2),
            second=1.6530,
            places=4,
        )

    def test_approx_euclidean(self) -> None:
        assert_almost_equal(
            first=approx_entropy(x=self.data_noise, order=3, metric="euclidean"),
            second=0.3923,
            places=4,
        )

    def test_approx_random(self) -> None:
        assert_almost_equal(
            first=approx_entropy(x=self.data_random),
            second=1.8177169675974412,
            places=4,
        )

    def test_sample_basic(self) -> None:
        assert_almost_equal(
            first=sample_entropy(x=self.data_airline),
            second=0.6177,
            places=4,
        )

    def test_sample_noise(self) -> None:
        assert_almost_equal(
            first=sample_entropy(x=self.data_noise, order=2),
            second=2.1540,
            places=4,
        )

    def test_sample_euclidean(self) -> None:
        assert_almost_equal(
            first=sample_entropy(x=self.data_noise, order=3, metric="euclidean"),
            second=2.7265,
            places=4,
        )

    def test_sample_random(self) -> None:
        assert_almost_equal(
            first=sample_entropy(x=self.data_random),
            second=2.2017103266074773,
            places=4,
        )

    def test_sample_sine(self) -> None:
        assert_almost_equal(
            first=sample_entropy(x=self.data_sine),
            second=0.1633,
            places=4,
        )

    def test_sample_line(self) -> None:
        assert_almost_equal(
            first=sample_entropy(x=self.data_line),
            second=0.0000,
            places=4,
        )

    def test_permutation_basic(self) -> None:
        assert_almost_equal(
            first=permutation_entropy(x=self.data_airline),
            second=2.3601,
            places=4,
        )

    def test_permutation_simple(self) -> None:
        assert_almost_equal(
            first=permutation_entropy(x=self.data_basic, order=2),
            second=0.9183,
            places=4,
        )

    def test_permutation_normalised(self) -> None:
        assert_almost_equal(
            first=permutation_entropy(x=self.data_basic, order=2, normalize=True),
            second=0.9183,
            places=4,
        )

    def test_permutation_gaussian_noise(self) -> None:
        assert_almost_equal(
            first=permutation_entropy(x=self.data_noise, order=2),
            second=0.998536,
            places=6,
        )

    def test_permutation_normalised_noise(self) -> None:
        assert_almost_equal(
            first=permutation_entropy(x=self.data_noise, order=2, normalize=True),
            second=0.998536,
            places=6,
        )

    def test_permutation_multiple_delays(self) -> None:
        assert_almost_equal(
            first=permutation_entropy(x=self.data_noise, delay=[1, 2, 3], normalize=True),
            second=0.9982,
            places=4,
        )

    def test_permutation_random(self) -> None:
        assert_almost_equal(
            first=permutation_entropy(x=self.data_random, normalize=True),
            second=0.9996512500846719,
            places=4,
        )

    def test_permutation_sine(self) -> None:
        assert_almost_equal(
            first=permutation_entropy(x=self.data_sine, normalize=True),
            second=0.4463,
            places=4,
        )

    def test_permutation_line(self) -> None:
        assert_almost_equal(
            first=permutation_entropy(x=self.data_line, normalize=True),
            second=0.0000,
            places=4,
        )

    def test_spectral_basic(self) -> None:
        assert_almost_equal(
            first=spectral_entropy(x=self.data_airline, sf=12),
            second=2.6538,
            places=4,
        )

    def test_spectral_sine(self) -> None:
        assert_almost_equal(
            first=spectral_entropy(x=self.data_sine, sf=100, method="fft"),
            second=6.2329e-27,
            places=4,
        )

    def test_spectral_welch(self) -> None:
        assert_almost_equal(
            first=spectral_entropy(x=self.data_sine, sf=100, method="welch"),
            second=1.2924,
            places=4,
        )

    def test_spectral_normalised(self) -> None:
        assert_almost_equal(
            first=spectral_entropy(x=self.data_sine, sf=100, method="fft", normalize=True),
            second=5.9070e-28,
            places=4,
        )

    def test_spectral_2d(self) -> None:
        assert list(spectral_entropy(x=self.data_2d, sf=100, normalize=True).round(4)) == [
            0.9445,
            0.9437,
            0.9450,
            0.9436,
        ]

    def test_spectral_noise(self) -> None:
        assert_almost_equal(
            first=spectral_entropy(x=self.data_noise, sf=100, normalize=True),
            second=0.9400,
            places=4,
        )

    def test_svd_entropy_direct(self) -> None:
        result = svd_entropy(x=self.data_random)
        assert isinstance(result, float)

    def test_entropy_algorithms_perm_svd(self) -> None:
        res_perm = entropy(x=self.data_random, algorithm="perm")
        res_svd = entropy(x=self.data_random, algorithm="svd")
        assert isinstance(res_perm, float)
        assert isinstance(res_svd, float)

    def test_is_regular_with_none_tolerance(self) -> None:
        # Covers part of is_regular tolerance logic
        result = is_regular(x=self.data_random, tolerance=None)
        assert isinstance(result, dict)
        assert "result" in result

    def test_is_regular_with_default_tolerance_string(self) -> None:
        # Covers part of is_regular tolerance logic
        result = is_regular(x=self.data_random, tolerance="default")
        assert isinstance(result, dict)
        assert "result" in result
