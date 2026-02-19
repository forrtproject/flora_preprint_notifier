"""Tests for osf_sync.dynamo.suppression_repo — DynamoDB suppression table."""
import unittest
from unittest.mock import MagicMock, Mock, patch

from botocore.exceptions import ClientError

from osf_sync.dynamo.suppression_repo import SuppressionRepo


def _make_repo() -> tuple:
    """Create a SuppressionRepo with a mocked DynamoDB table."""
    with patch("osf_sync.dynamo.suppression_repo.get_dynamo_resource") as mock_ddb:
        table = MagicMock()
        mock_ddb.return_value.Table.return_value = table
        repo = SuppressionRepo()
    return repo, table


class TestSuppressionRepo(unittest.TestCase):
    def test_is_suppressed_returns_true_when_item_exists(self) -> None:
        repo, table = _make_repo()
        table.get_item.return_value = {"Item": {"email": "bob@example.com", "reason": "bounce"}}
        self.assertTrue(repo.is_suppressed("Bob@Example.com"))
        table.get_item.assert_called_once_with(Key={"email": "bob@example.com"})

    def test_is_suppressed_returns_false_when_no_item(self) -> None:
        repo, table = _make_repo()
        table.get_item.return_value = {}
        self.assertFalse(repo.is_suppressed("unknown@example.com"))

    def test_is_suppressed_returns_false_for_empty_input(self) -> None:
        repo, table = _make_repo()
        self.assertFalse(repo.is_suppressed(""))
        self.assertFalse(repo.is_suppressed("  "))
        table.get_item.assert_not_called()

    def test_add_suppression_returns_true_when_new(self) -> None:
        repo, table = _make_repo()
        table.put_item.return_value = {}
        result = repo.add_suppression("Alice@Uni.Edu", "bounce")
        self.assertTrue(result)
        call_kwargs = table.put_item.call_args.kwargs
        item = call_kwargs["Item"]
        self.assertEqual(item["email"], "alice@uni.edu")
        self.assertEqual(item["reason"], "bounce")
        self.assertIn("suppressed_at", item)

    def test_add_suppression_returns_false_when_already_exists(self) -> None:
        repo, table = _make_repo()
        error_response = {"Error": {"Code": "ConditionalCheckFailedException", "Message": "exists"}}
        table.put_item.side_effect = ClientError(error_response, "PutItem")
        table.meta.client.exceptions.ConditionalCheckFailedException = type(
            "ConditionalCheckFailedException", (ClientError,), {}
        )
        # Re-raise as the correct exception type
        table.put_item.side_effect = table.meta.client.exceptions.ConditionalCheckFailedException(
            error_response, "PutItem"
        )
        result = repo.add_suppression("alice@uni.edu", "bounce")
        self.assertFalse(result)

    def test_add_suppression_returns_false_for_empty_input(self) -> None:
        repo, table = _make_repo()
        self.assertFalse(repo.add_suppression("", "bounce"))
        table.put_item.assert_not_called()

    def test_remove_suppression_calls_delete(self) -> None:
        repo, table = _make_repo()
        repo.remove_suppression("Alice@Uni.Edu")
        table.delete_item.assert_called_once_with(Key={"email": "alice@uni.edu"})

    def test_remove_suppression_noop_for_empty(self) -> None:
        repo, table = _make_repo()
        repo.remove_suppression("")
        table.delete_item.assert_not_called()

    def test_list_suppressions_returns_items(self) -> None:
        repo, table = _make_repo()
        table.scan.return_value = {
            "Items": [
                {"email": "a@x.com", "reason": "bounce"},
                {"email": "b@y.com", "reason": "unsubscribe"},
            ]
        }
        items = repo.list_suppressions()
        self.assertEqual(len(items), 2)

    def test_list_suppressions_handles_pagination(self) -> None:
        repo, table = _make_repo()
        table.scan.side_effect = [
            {"Items": [{"email": "a@x.com"}], "LastEvaluatedKey": {"email": "a@x.com"}},
            {"Items": [{"email": "b@y.com"}]},
        ]
        items = repo.list_suppressions()
        self.assertEqual(len(items), 2)
        self.assertEqual(table.scan.call_count, 2)


if __name__ == "__main__":
    unittest.main()
