"""Tests for osf_sync.email.suppression — dual-source (blacklist + DynamoDB) checking."""
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from osf_sync.email.blacklist import clear_blacklist_cache
from osf_sync.email.suppression import is_suppressed


class TestDualSuppression(unittest.TestCase):
    def setUp(self) -> None:
        clear_blacklist_cache()

    def tearDown(self) -> None:
        clear_blacklist_cache()

    def test_blacklisted_returns_true_even_if_ddb_says_no(self) -> None:
        """Blacklist hit short-circuits — DynamoDB is not even consulted."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bl.txt")
            with open(path, "w") as fh:
                fh.write("email:blocked@uni.edu\n")

            with patch.dict(os.environ, {"EMAIL_BLACKLIST_PATH": path}):
                clear_blacklist_cache()
                repo = Mock()
                repo.is_suppressed.return_value = False
                self.assertTrue(is_suppressed("blocked@uni.edu", repo=repo))
                # DDB should not have been consulted
                repo.is_suppressed.assert_not_called()

    def test_ddb_suppressed_returns_true_when_blacklist_says_no(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bl.txt")
            with open(path, "w") as fh:
                fh.write("# empty blacklist\n")

            with patch.dict(os.environ, {"EMAIL_BLACKLIST_PATH": path}):
                clear_blacklist_cache()
                repo = Mock()
                repo.is_suppressed.return_value = True
                self.assertTrue(is_suppressed("bounced@uni.edu", repo=repo))
                repo.is_suppressed.assert_called_once_with("bounced@uni.edu")

    def test_neither_suppressed_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bl.txt")
            with open(path, "w") as fh:
                fh.write("# empty blacklist\n")

            with patch.dict(os.environ, {"EMAIL_BLACKLIST_PATH": path}):
                clear_blacklist_cache()
                repo = Mock()
                repo.is_suppressed.return_value = False
                self.assertFalse(is_suppressed("ok@uni.edu", repo=repo))

    def test_fail_open_on_ddb_error(self) -> None:
        """DynamoDB errors should not block sending (fail-open)."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bl.txt")
            with open(path, "w") as fh:
                fh.write("# empty blacklist\n")

            with patch.dict(os.environ, {"EMAIL_BLACKLIST_PATH": path}):
                clear_blacklist_cache()
                repo = Mock()
                repo.is_suppressed.side_effect = Exception("DynamoDB timeout")
                self.assertFalse(is_suppressed("ok@uni.edu", repo=repo))

    def test_empty_email_is_suppressed(self) -> None:
        self.assertTrue(is_suppressed(""))
        self.assertTrue(is_suppressed("  "))


if __name__ == "__main__":
    unittest.main()
