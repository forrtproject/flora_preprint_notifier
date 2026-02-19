import unittest
from types import SimpleNamespace
from unittest.mock import patch

from osf_sync import upsert


class UpsertWindowTests(unittest.TestCase):
    def test_filter_uses_creation_date_not_publication_date(self) -> None:
        rows = [
            {
                "id": "p1",
                "attributes": {
                    "date_created": "2026-03-01T00:00:00Z",
                    "date_published": "2027-01-01T00:00:00Z",
                },
                "links": {},
            }
        ]
        cfg = SimpleNamespace(ingest=SimpleNamespace(anchor_date="2026-02-20"))
        with patch.object(upsert, "RUNTIME_CONFIG", cfg):
            kept, skipped_date, _skipped_links, _records = upsert._filter_ingest_rows(rows)

        self.assertEqual(len(kept), 1)
        self.assertEqual(skipped_date, 0)

    def test_filter_applies_plus_minus_six_month_window(self) -> None:
        rows = [
            {
                "id": "before_window",
                "attributes": {"date_created": "2025-08-19T00:00:00Z"},
                "links": {},
            },
            {
                "id": "inside_window",
                "attributes": {"date_created": "2026-08-20T00:00:00Z"},
                "links": {},
            },
            {
                "id": "after_window",
                "attributes": {"date_created": "2026-08-21T00:00:00Z"},
                "links": {},
            },
        ]
        cfg = SimpleNamespace(ingest=SimpleNamespace(anchor_date="2026-02-20"))
        with patch.object(upsert, "RUNTIME_CONFIG", cfg):
            kept, skipped_date, _skipped_links, _records = upsert._filter_ingest_rows(rows)

        kept_ids = {row["id"] for row in kept}
        self.assertEqual(kept_ids, {"inside_window"})
        self.assertEqual(skipped_date, 2)


if __name__ == "__main__":
    unittest.main()
