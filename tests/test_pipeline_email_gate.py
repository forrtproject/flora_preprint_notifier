import unittest
from types import SimpleNamespace
from unittest.mock import patch

from osf_sync import pipeline


class PipelineEmailGateTests(unittest.TestCase):
    def test_run_stage_email_is_blocked_by_default(self) -> None:
        args = SimpleNamespace(
            stage="email",
            limit=5,
            max_seconds=60,
            spread_seconds=None,
            dry_run=False,
            osf_id=None,
        )
        with patch("osf_sync.pipeline._email_stage_enabled", return_value=False), patch(
            "osf_sync.pipeline.process_email_batch"
        ) as mock_email:
            out = pipeline.run_stage(args)
        mock_email.assert_not_called()
        self.assertTrue(out.get("skipped_disabled"))
        self.assertEqual(out.get("processed"), 0)

    def test_run_post_grobid_skips_email_when_disabled(self) -> None:
        args = SimpleNamespace(
            include_email=True,
            email_limit=10,
            extract_limit=0,
            enrich_limit=0,
            threshold=None,
            mailto=None,
            osf_id=None,
            ref_id=None,
            debug=False,
            enrich_workers=1,
            max_seconds_per_stage=None,
            owner=None,
            lease_seconds=1800,
            dry_run=True,
            limit_lookup=0,
            limit_screen=0,
            cache_ttl_hours=None,
            no_persist=False,
            include_checked=False,
            skip_author=True,
            skip_randomization=True,
        )

        with patch("osf_sync.pipeline._email_stage_enabled", return_value=False), patch(
            "osf_sync.pipeline.process_extract_batch", return_value={"stage": "extract", "processed": 0, "failed": 0}
        ), patch(
            "osf_sync.pipeline.process_enrich_batch", return_value={"stage": "enrich", "checked": 0, "updated": 0, "failed": 0}
        ), patch(
            "osf_sync.pipeline.process_flora_batch", return_value={"stage": "flora", "lookup": {}, "screen": []}
        ), patch(
            "osf_sync.pipeline.process_email_batch"
        ) as mock_email, patch(
            "osf_sync.pipeline._notify_pipeline_summary"
        ):
            out = pipeline.run_post_grobid(args)

        mock_email.assert_not_called()
        self.assertIn("email", out["stages"])
        self.assertTrue(out["stages"]["email"].get("skipped_disabled"))


if __name__ == "__main__":
    unittest.main()
