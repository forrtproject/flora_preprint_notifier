import unittest
from unittest.mock import patch

from botocore.exceptions import ClientError

from osf_sync.dynamo.preprints_repo import PreprintsRepo


class _FakeTable:
    def __init__(self, name: str) -> None:
        self.name = name
        self.put_calls = []
        self.update_calls = []
        self.items = {}

    def put_item(self, **kwargs):
        item = dict(kwargs.get("Item") or {})
        cond = kwargs.get("ConditionExpression")
        osf_id = item.get("osf_id")
        if cond == "attribute_not_exists(osf_id)" and osf_id in self.items:
            raise ClientError(
                {"Error": {"Code": "ConditionalCheckFailedException", "Message": "exists"}},
                "PutItem",
            )
        if osf_id:
            self.items[osf_id] = item
        self.put_calls.append(kwargs)
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def update_item(self, **kwargs):
        self.update_calls.append(kwargs)
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def get_item(self, **kwargs):
        key = kwargs.get("Key") or {}
        osf_id = key.get("osf_id")
        if osf_id in self.items:
            return {"Item": self.items[osf_id]}
        return {}


class _FakeDynamo:
    def __init__(self) -> None:
        self._tables = {}

    def Table(self, name: str):
        if name not in self._tables:
            self._tables[name] = _FakeTable(name)
        return self._tables[name]


class PreprintExclusionTests(unittest.TestCase):
    @patch("osf_sync.dynamo.preprints_repo.get_dynamo_resource")
    def test_mark_preprint_excluded_inserts_single_row_per_osf_id(self, mock_get) -> None:
        fake = _FakeDynamo()
        mock_get.return_value = fake
        repo = PreprintsRepo()

        first = repo.mark_preprint_excluded(
            osf_id="osf123",
            reason="unsupported_file_format",
            stage="pdf",
            details={"provider_id": "psyarxiv"},
        )
        second = repo.mark_preprint_excluded(
            osf_id="osf123",
            reason="no_references_extracted",
            stage="extract",
        )

        self.assertTrue(first)
        self.assertFalse(second)

        excluded_table = fake._tables[repo.t_excluded.name]
        self.assertEqual(len(excluded_table.put_calls), 1)
        self.assertEqual(len(excluded_table.items), 1)

        item = excluded_table.items["osf123"]
        self.assertEqual(item["osf_id"], "osf123")
        self.assertEqual(item["exclusion_reason"], "unsupported_file_format")
        self.assertEqual(item["exclusion_stage"], "pdf")

    @patch("osf_sync.dynamo.preprints_repo.get_dynamo_resource")
    def test_mark_preprint_excluded_requires_reason_and_osf_id(self, mock_get) -> None:
        fake = _FakeDynamo()
        mock_get.return_value = fake
        repo = PreprintsRepo()

        with self.assertRaises(ValueError):
            repo.mark_preprint_excluded(osf_id="", reason="x")
        with self.assertRaises(ValueError):
            repo.mark_preprint_excluded(osf_id="abc", reason=" ")


if __name__ == "__main__":
    unittest.main()
