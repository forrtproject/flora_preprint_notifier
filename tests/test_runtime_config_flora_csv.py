import tempfile
import unittest
from pathlib import Path

from osf_sync.runtime_config import load_runtime_config


class RuntimeConfigFloraCsvTests(unittest.TestCase):
    def test_loads_flora_csv_settings_from_runtime_toml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runtime.toml"
            path.write_text(
                "\n".join(
                    [
                        "[flora]",
                        'csv_url = "https://example.org/flora.csv"',
                        'csv_path = "tmp/flora.csv"',
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_runtime_config(path)

        self.assertEqual(cfg.flora.csv_url, "https://example.org/flora.csv")
        self.assertEqual(cfg.flora.csv_path, "tmp/flora.csv")

    def test_defaults_flora_csv_settings_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "runtime.toml"
            path.write_text("[flora]\n", encoding="utf-8")

            cfg = load_runtime_config(path)

        self.assertIn("FReD-data", cfg.flora.csv_url)
        self.assertEqual(cfg.flora.csv_path, "data/flora.csv")


if __name__ == "__main__":
    unittest.main()
