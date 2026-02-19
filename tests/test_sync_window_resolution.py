import datetime as dt
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from osf_sync import pipeline


class SyncWindowResolutionTests(unittest.TestCase):
    def test_prod_anchor_window_used_when_cursor_missing(self) -> None:
        cfg = SimpleNamespace(ingest=SimpleNamespace(anchor_date="2026-02-20", window_months=6))
        now_utc = dt.datetime(2026, 4, 1, tzinfo=dt.timezone.utc)
        with patch.dict("os.environ", {"PIPELINE_ENV": "prod"}, clear=False), patch.object(pipeline, "RUNTIME_CONFIG", cfg):
            since, until, mode = pipeline._resolve_sync_window(None, now_utc=now_utc)
        self.assertEqual(since, "2025-08-20")
        self.assertEqual(until, "2026-04-01")
        self.assertEqual(mode, "prod_anchor_window")

    def test_prod_cursor_is_clamped_to_anchor_window_start(self) -> None:
        cfg = SimpleNamespace(ingest=SimpleNamespace(anchor_date="2026-02-20", window_months=6))
        now_utc = dt.datetime(2026, 4, 1, tzinfo=dt.timezone.utc)
        cursor = dt.datetime(2025, 1, 15, tzinfo=dt.timezone.utc)
        with patch.dict("os.environ", {"PIPELINE_ENV": "prod"}, clear=False), patch.object(pipeline, "RUNTIME_CONFIG", cfg):
            since, until, mode = pipeline._resolve_sync_window(cursor, now_utc=now_utc)
        self.assertEqual(since, "2025-08-20")
        self.assertEqual(until, "2026-04-01")
        self.assertEqual(mode, "prod_anchor_window")

    def test_dev_uses_7_day_window_when_cursor_missing(self) -> None:
        cfg = SimpleNamespace(ingest=SimpleNamespace(anchor_date="2026-02-20", window_months=6))
        now_utc = dt.datetime(2026, 2, 18, tzinfo=dt.timezone.utc)
        with patch.dict("os.environ", {"PIPELINE_ENV": "dev"}, clear=False), patch.object(pipeline, "RUNTIME_CONFIG", cfg):
            since, until, mode = pipeline._resolve_sync_window(None, now_utc=now_utc)
        self.assertEqual(since, "2026-02-11")
        self.assertIsNone(until)
        self.assertEqual(mode, "dev_recent")

    def test_dev_cursor_is_clamped_to_recent_window(self) -> None:
        cfg = SimpleNamespace(ingest=SimpleNamespace(anchor_date="2026-02-20", window_months=6))
        now_utc = dt.datetime(2026, 2, 18, tzinfo=dt.timezone.utc)
        cursor = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
        with patch.dict("os.environ", {"PIPELINE_ENV": "dev"}, clear=False), patch.object(pipeline, "RUNTIME_CONFIG", cfg):
            since, until, mode = pipeline._resolve_sync_window(cursor, now_utc=now_utc)
        self.assertEqual(since, "2026-02-11")
        self.assertIsNone(until)
        self.assertEqual(mode, "dev_recent")

    def test_dev_override_start_date_is_used(self) -> None:
        cfg = SimpleNamespace(ingest=SimpleNamespace(anchor_date=None, window_months=6))
        with patch.dict(
            "os.environ",
            {"PIPELINE_ENV": "dev", "SYNC_START_DATE_OVERRIDE": "2025-01-01"},
            clear=False,
        ), patch.object(pipeline, "RUNTIME_CONFIG", cfg):
            since, until, mode = pipeline._resolve_sync_window(None)
        self.assertEqual(since, "2025-01-01")
        self.assertIsNone(until)
        self.assertEqual(mode, "dev_override")

    def test_prod_override_defaults_end_to_anchor(self) -> None:
        cfg = SimpleNamespace(ingest=SimpleNamespace(anchor_date="2026-03-15", window_months=6))
        now_utc = dt.datetime(2026, 4, 1, tzinfo=dt.timezone.utc)
        with patch.dict(
            "os.environ",
            {"PIPELINE_ENV": "prod", "SYNC_START_DATE_OVERRIDE": "2026-03-01"},
            clear=False,
        ), patch.object(pipeline, "RUNTIME_CONFIG", cfg):
            since, until, mode = pipeline._resolve_sync_window(None, now_utc=now_utc)
        self.assertEqual(since, "2026-03-01")
        self.assertEqual(until, "2026-04-01")
        self.assertEqual(mode, "prod_override")

    def test_override_start_after_end_raises(self) -> None:
        cfg = SimpleNamespace(ingest=SimpleNamespace(anchor_date="2026-03-15", window_months=6))
        with patch.dict(
            "os.environ",
            {
                "PIPELINE_ENV": "prod",
                "SYNC_START_DATE_OVERRIDE": "2026-03-20",
                "SYNC_END_DATE_OVERRIDE": "2026-03-10",
            },
            clear=False,
        ), patch.object(pipeline, "RUNTIME_CONFIG", cfg):
            with self.assertRaises(RuntimeError):
                pipeline._resolve_sync_window(None)

    def test_prod_requires_anchor_date(self) -> None:
        cfg = SimpleNamespace(ingest=SimpleNamespace(anchor_date=None, window_months=6))
        with patch.dict("os.environ", {"PIPELINE_ENV": "prod"}, clear=False), patch.object(pipeline, "RUNTIME_CONFIG", cfg):
            with self.assertRaises(RuntimeError):
                pipeline._resolve_sync_window(None)

    def test_should_write_cursor_defaults_false_for_override(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(pipeline._should_write_cursor("prod_override"))

    def test_should_write_cursor_override_enabled_by_env(self) -> None:
        with patch.dict("os.environ", {"SYNC_OVERRIDE_WRITES_CURSOR": "true"}, clear=True):
            self.assertTrue(pipeline._should_write_cursor("prod_override"))

    def test_should_write_cursor_can_be_globally_disabled(self) -> None:
        with patch.dict(
            "os.environ",
            {"SYNC_OVERRIDE_WRITES_CURSOR": "true", "SYNC_DISABLE_CURSOR_WRITE": "true"},
            clear=True,
        ):
            self.assertFalse(pipeline._should_write_cursor("prod_anchor_window"))

    def test_ingest_config_changed_detects_anchor_change(self) -> None:
        cfg = SimpleNamespace(ingest=SimpleNamespace(anchor_date="2026-03-15", window_months=6, backfill_on_config_change=True))
        meta = {"anchor_date": "2026-03-10", "window_months": 6}
        with patch.object(pipeline, "RUNTIME_CONFIG", cfg):
            self.assertTrue(pipeline._ingest_config_changed(meta))

    def test_ingest_config_changed_detects_window_change(self) -> None:
        cfg = SimpleNamespace(ingest=SimpleNamespace(anchor_date="2026-03-15", window_months=6, backfill_on_config_change=True))
        meta = {"anchor_date": "2026-03-15", "window_months": 5}
        with patch.object(pipeline, "RUNTIME_CONFIG", cfg):
            self.assertTrue(pipeline._ingest_config_changed(meta))

    def test_ingest_config_changed_false_when_same(self) -> None:
        cfg = SimpleNamespace(ingest=SimpleNamespace(anchor_date="2026-03-15", window_months=6, backfill_on_config_change=True))
        meta = {"anchor_date": "2026-03-15", "window_months": 6}
        with patch.object(pipeline, "RUNTIME_CONFIG", cfg):
            self.assertFalse(pipeline._ingest_config_changed(meta))

    def test_ingest_config_changed_false_when_disabled(self) -> None:
        cfg = SimpleNamespace(ingest=SimpleNamespace(anchor_date="2026-03-15", window_months=6, backfill_on_config_change=False))
        meta = {"anchor_date": "2026-03-10", "window_months": 5}
        with patch.object(pipeline, "RUNTIME_CONFIG", cfg):
            self.assertFalse(pipeline._ingest_config_changed(meta))


if __name__ == "__main__":
    unittest.main()
