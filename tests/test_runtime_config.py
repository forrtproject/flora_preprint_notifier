import tempfile
import unittest
from pathlib import Path

from osf_sync.runtime_config import load_runtime_config


class RuntimeConfigTests(unittest.TestCase):
    def test_email_progress_emails_defaults_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.toml"
            path.write_text("[email]\n", encoding="utf-8")
            cfg = load_runtime_config(path)
        self.assertTrue(cfg.email.progress_emails)

    def test_email_progress_emails_reads_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.toml"
            path.write_text("[email]\nprogress_emails = false\n", encoding="utf-8")
            cfg = load_runtime_config(path)
        self.assertFalse(cfg.email.progress_emails)


if __name__ == "__main__":
    unittest.main()
