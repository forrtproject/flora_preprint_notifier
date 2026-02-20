import unittest
from types import SimpleNamespace
from unittest.mock import patch

from osf_sync import pipeline


def _stage_result(stage: str, *, processed: int, failed: int = 0, skipped_claimed: int = 0):
    return {
        "stage": stage,
        "selected": processed + failed,
        "claimed": processed + failed,
        "processed": processed,
        "failed": failed,
        "skipped_claimed": skipped_claimed,
        "stopped_due_to_time": False,
        "dry_run": False,
    }


class GrobidInterleaveTests(unittest.TestCase):
    @patch("osf_sync.pipeline._notify_pipeline_summary")
    @patch("osf_sync.pipeline.process_grobid_batch")
    @patch("osf_sync.pipeline.process_pdf_batch")
    @patch("osf_sync.pipeline.sync_from_osf")
    def test_interleave_keeps_going_after_transient_empty_grobid(
        self,
        mock_sync,
        mock_pdf,
        mock_grobid,
        _mock_notify,
    ) -> None:
        mock_sync.return_value = {"upserted": 100, "cursor": "2026-01-01T00:00:00+00:00", "dry_run": False}
        mock_pdf.side_effect = [
            _stage_result("pdf", processed=50),
            _stage_result("pdf", processed=0),
        ]
        mock_grobid.side_effect = [
            _stage_result("grobid", processed=0),   # transient empty round
            _stage_result("grobid", processed=50),  # work appears next round
            _stage_result("grobid", processed=0),
            _stage_result("grobid", processed=0),
            _stage_result("grobid", processed=0),   # idle threshold reached
        ]

        args = SimpleNamespace(
            max_seconds=None,
            max_seconds_sync=1200,
            subject=None,
            batch_size=100,
            sync_limit=1000,
            pdf_limit=1000,
            grobid_limit=1000,
            interleave_batch=50,
            download_workers=1,
            owner="test",
            lease_seconds=30,
            dry_run=False,
        )

        out = pipeline.run_grobid_stages(args)

        self.assertEqual(out["stages"]["pdf"]["processed"], 50)
        self.assertEqual(out["stages"]["grobid"]["processed"], 50)
        self.assertGreaterEqual(mock_grobid.call_count, 2)


if __name__ == "__main__":
    unittest.main()
