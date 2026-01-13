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


class TestErrors(unittest.TestCase):

    def test_generate_error_message(self) -> None:
        options = {"key1": ["val1", "val2"], "key2": ["val3"]}
        msg = generate_error_message("param", "bad_val", options)
        assert "Invalid option for `param` parameter: 'bad_val'" in msg
        assert "For the 'key1' option, use one of: '['val1', 'val2']'" in msg

    def test_is_almost_equal(self) -> None:
        assert is_almost_equal(1.0, 1.0)
        assert is_almost_equal(1.0, 1.00000001, places=7)
        assert not is_almost_equal(1.0, 1.1)
        assert is_almost_equal(1.0, 1.05, delta=0.1)
        assert not is_almost_equal(1.0, 1.05, delta=0.01)

        with raises(ValueError):
            is_almost_equal(1.0, 1.1, places=7, delta=0.1)

    def test_assert_almost_equal(self):
        assert_almost_equal(1.0, 1.0)
        assert_almost_equal(1.0, 1.00000001, places=7)
        assert_almost_equal(1.0, 1.05, delta=0.1)

        with raises(AssertionError):
            assert_almost_equal(1.0, 1.1)

        with raises(AssertionError):
            assert_almost_equal(1.0, 1.05, delta=0.01)

        with raises(AssertionError, match="custom"):
            assert_almost_equal(1.0, 1.1, msg="custom")
