# ---------------------------------------------------------------------------- #
#                                                                              #
#    Setup                                                                  ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Imports                                                                   ####
# ---------------------------------------------------------------------------- #


# ## Python StdLib Imports ----
import unittest

# ## Python Third Party Imports ----
from pytest import raises

# ## Local First Party Imports ----
from ts_stat_tests.utils.errors import (
    assert_almost_equal,
    generate_error_message,
    is_almost_equal,
)


# ---------------------------------------------------------------------------- #
#                                                                              #
#    Testing                                                                ####
#                                                                              #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Tests                                                                     ####
# ---------------------------------------------------------------------------- #


class TestErrors(unittest.TestCase):

    def test_generate_error_message(self) -> None:
        options = {"key1": ["val1", "val2"], "key2": ["val3"]}
        msg = generate_error_message("param", "bad_val", options)
        assert "Invalid 'param'" in msg
        assert "bad_val" in msg
        assert "['val1', 'val2']" in msg
        assert "'key2': ['val3']" in msg

    def test_is_almost_equal(self) -> None:
        assert is_almost_equal(1.0, 1.0)
        assert is_almost_equal(1.0, 1.00000001, places=7)
        assert not is_almost_equal(1.0, 1.1)
        assert is_almost_equal(1.0, 1.05, delta=0.1)
        assert not is_almost_equal(1.0, 1.05, delta=0.01)

    def test_assert_almost_equal(self) -> None:
        assert_almost_equal(1.0, 1.0)
        assert_almost_equal(1.0, 1.00000001, places=7)
        assert_almost_equal(1.0, 1.05, delta=0.1)

        with raises(AssertionError):
            assert_almost_equal(1.0, 1.1)

        with raises(AssertionError):
            assert_almost_equal(1.0, 1.05, delta=0.01)

        with raises(AssertionError, match="custom"):
            assert_almost_equal(1.0, 1.1, msg="custom")

    def test_is_almost_equal_value_error(self) -> None:
        with raises(ValueError, match="Specify `delta` or `places`, not both."):
            is_almost_equal(1.0, 1.1, places=7, delta=0.1)

    def test_is_almost_equal_with_delta(self) -> None:
        assert is_almost_equal(1.0, 1.05, delta=0.1) is True
        assert is_almost_equal(1.0, 1.2, delta=0.1) is False

    def test_assert_almost_equal_failures_1(self) -> None:
        with raises(AssertionError) as e:
            assert_almost_equal(1.0, 1.2, delta=0.1)
        assert "1.0 != 1.2" in str(e.value)
        assert "delta=0.1" in str(e.value)

    def test_assert_almost_equal_failures_2(self) -> None:
        with raises(AssertionError) as e:
            assert_almost_equal(1.0, 1.0000001, places=10)
        assert "1.0 != 1.0000001" in str(e.value)
        assert "places=10" in str(e.value)

    def test_assert_almost_equal_failures_3(self) -> None:
        with raises(AssertionError) as e:
            assert_almost_equal(1.0, 2.0, msg="Custom error message")
        assert "Custom error message" in str(e.value)
