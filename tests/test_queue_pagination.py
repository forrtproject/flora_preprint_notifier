import unittest

from osf_sync.dynamo.preprints_repo import PreprintsRepo


class _FakeQueueTable:
    def __init__(self, *, query_pages=None, scan_pages=None, raise_query: bool = False) -> None:
        self.query_pages = list(query_pages or [])
        self.scan_pages = list(scan_pages or [])
        self.raise_query = raise_query
        self.query_calls = []
        self.scan_calls = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        if self.raise_query:
            raise RuntimeError("query unavailable")
        if self.query_pages:
            return self.query_pages.pop(0)
        return {"Items": []}

    def scan(self, **kwargs):
        self.scan_calls.append(kwargs)
        if self.scan_pages:
            return self.scan_pages.pop(0)
        return {"Items": []}


class QueuePaginationTests(unittest.TestCase):
    def test_select_for_grobid_paginates_until_filtered_items_found(self) -> None:
        table = _FakeQueueTable(
            query_pages=[
                {"Items": [], "LastEvaluatedKey": {"queue_grobid": "pending", "osf_id": "k1"}},
                {"Items": [{"osf_id": "a1"}]},
            ]
        )
        repo = PreprintsRepo.__new__(PreprintsRepo)
        repo.t_preprints = table

        out = PreprintsRepo.select_for_grobid(repo, limit=1)

        self.assertEqual(out, ["a1"])
        self.assertEqual(len(table.query_calls), 2)
        self.assertNotIn("ExclusiveStartKey", table.query_calls[0])
        self.assertIn("ExclusiveStartKey", table.query_calls[1])

    def test_select_for_grobid_not_capped_to_five_pages(self) -> None:
        query_pages = []
        for i in range(6):
            query_pages.append(
                {
                    "Items": [],
                    "LastEvaluatedKey": {"queue_grobid": "pending", "osf_id": f"k{i}"},
                }
            )
        query_pages.append({"Items": [{"osf_id": "late"}]})

        table = _FakeQueueTable(query_pages=query_pages)
        repo = PreprintsRepo.__new__(PreprintsRepo)
        repo.t_preprints = table

        out = PreprintsRepo.select_for_grobid(repo, limit=1)

        self.assertEqual(out, ["late"])
        self.assertEqual(len(table.query_calls), 7)

    def test_select_for_pdf_fallback_scan_paginates(self) -> None:
        table = _FakeQueueTable(
            raise_query=True,
            scan_pages=[
                {"Items": [], "LastEvaluatedKey": {"osf_id": "s1"}},
                {"Items": [{"osf_id": "p1"}]},
            ],
        )
        repo = PreprintsRepo.__new__(PreprintsRepo)
        repo.t_preprints = table

        out = PreprintsRepo.select_for_pdf(repo, limit=1)

        self.assertEqual(out, ["p1"])
        self.assertEqual(len(table.scan_calls), 2)


if __name__ == "__main__":
    unittest.main()
