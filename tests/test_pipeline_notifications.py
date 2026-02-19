import unittest
from types import SimpleNamespace
from unittest.mock import patch

from osf_sync import pipeline


class PipelineNotificationTests(unittest.TestCase):
    @patch("osf_sync.email.gmail.send_email")
    def test_notify_skips_when_progress_emails_disabled(self, mock_send_email) -> None:
        cfg = SimpleNamespace(email=SimpleNamespace(progress_emails=False))
        result = {"stages": {"grobid": {"processed": 3, "failed": 0, "stopped_due_to_time": False}}}
        with patch.object(pipeline, "RUNTIME_CONFIG", cfg), patch.object(
            pipeline, "PIPELINE_NOTIFY_EMAIL", "ops@example.com"
        ):
            pipeline._notify_pipeline_summary(result)
        mock_send_email.assert_not_called()

    @patch("osf_sync.email.gmail.send_email")
    def test_notify_sends_when_progress_emails_enabled(self, mock_send_email) -> None:
        cfg = SimpleNamespace(email=SimpleNamespace(progress_emails=True))
        result = {"stages": {"grobid": {"processed": 3, "failed": 0, "stopped_due_to_time": False}}}
        with patch.object(pipeline, "RUNTIME_CONFIG", cfg), patch.object(
            pipeline, "PIPELINE_NOTIFY_EMAIL", "ops@example.com"
        ):
            pipeline._notify_pipeline_summary(result)
        mock_send_email.assert_called_once()


if __name__ == "__main__":
    unittest.main()
