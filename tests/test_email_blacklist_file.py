import os
import tempfile
import unittest
from unittest.mock import patch

from osf_sync.email.blacklist import clear_blacklist_cache, load_blacklist
from osf_sync.email.suppression import is_suppressed
from osf_sync.extraction.extract_author_list import _clean_email


class EmailBlacklistFileTests(unittest.TestCase):
    def test_default_blacklist_blocks_irb_local_part(self) -> None:
        self.assertIsNone(_clean_email("irb@uni.edu"))

    def test_file_backed_blacklist_used_for_suppression_and_cleaning(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "email_blacklist.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("local:contact\n")
                fh.write("domain:blocked.org\n")
                fh.write("email:exact@uni.edu\n")

            with patch.dict(os.environ, {"EMAIL_BLACKLIST_PATH": path}, clear=False):
                clear_blacklist_cache()
                bl = load_blacklist()
                self.assertIn("contact", bl["locals"])
                self.assertIn("blocked.org", bl["domains"])
                self.assertIn("exact@uni.edu", bl["emails"])

                self.assertTrue(is_suppressed("contact@x.edu"))
                self.assertTrue(is_suppressed("user@blocked.org"))
                self.assertTrue(is_suppressed("exact@uni.edu"))
                self.assertFalse(is_suppressed("ok@uni.edu"))

                self.assertIsNone(_clean_email("contact@x.edu"))
                self.assertIsNone(_clean_email("ok@blocked.org"))
                self.assertIsNone(_clean_email("exact@uni.edu"))
                self.assertEqual(_clean_email("ok@uni.edu"), "ok@uni.edu")

            clear_blacklist_cache()


if __name__ == "__main__":
    unittest.main()
