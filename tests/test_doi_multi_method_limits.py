import unittest
from unittest.mock import patch

from osf_sync.augmentation.doi_multi_method import enrich_missing_with_multi_method


class _FakeRepo:
    def __init__(self) -> None:
        self.preprint_selector_calls = 0
        self.select_refs_calls = []
        self.updated = []
        self.checked = []

    def select_osf_ids_with_refs_missing_doi(self, limit_preprints: int, *, skip_checked_within_seconds=None):
        self.preprint_selector_calls += 1
        assert limit_preprints == 3
        return ["p1", "p2", "p3"]

    def select_refs_missing_doi(
        self,
        limit,
        osf_id=None,
        *,
        ref_id=None,
        include_existing=False,
        skip_checked_within_seconds=None,
    ):
        self.select_refs_calls.append({"limit": limit, "osf_id": osf_id, "ref_id": ref_id})
        if osf_id == "p1":
            return [
                {"osf_id": "p1", "ref_id": "r1"},
                {"osf_id": "p1", "ref_id": "r2"},
            ]
        if osf_id == "p2":
            return [
                {"osf_id": "p2", "ref_id": "r3"},
            ]
        return []

    def update_reference_doi(self, osf_id: str, ref_id: str, doi: str, *, source: str) -> bool:
        self.updated.append((osf_id, ref_id, doi, source))
        return True

    def mark_reference_doi_checked(self, osf_id: str, ref_id: str) -> None:
        self.checked.append((osf_id, ref_id))


def _matched_row(ref, **kwargs):
    return {
        "status": "matched",
        "final_method": "crossref_title",
        "title_doi": f"10.1234/{ref['ref_id']}",
    }


class DoiMultiMethodLimitTests(unittest.TestCase):
    def test_limit_is_preprint_based_for_general_enrich(self) -> None:
        repo = _FakeRepo()
        with patch("osf_sync.augmentation.doi_multi_method.PreprintsRepo", return_value=repo), patch(
            "osf_sync.augmentation.doi_multi_method.process_reference", side_effect=_matched_row
        ):
            out = enrich_missing_with_multi_method(limit=2, workers=1)

        self.assertEqual(out["preprints_selected"], 2)
        self.assertTrue(out["limit_reached"])
        self.assertEqual(out["checked"], 3)
        self.assertEqual(out["updated"], 3)
        self.assertEqual(repo.preprint_selector_calls, 1)
        self.assertEqual([c["osf_id"] for c in repo.select_refs_calls], ["p1", "p2"])

    def test_osf_scoped_enrich_keeps_direct_reference_selection(self) -> None:
        repo = _FakeRepo()

        def _scoped_rows(limit, osf_id=None, **kwargs):
            repo.select_refs_calls.append({"limit": limit, "osf_id": osf_id})
            return [{"osf_id": "p9", "ref_id": "r9"}]

        repo.select_refs_missing_doi = _scoped_rows  # type: ignore[method-assign]

        with patch("osf_sync.augmentation.doi_multi_method.PreprintsRepo", return_value=repo), patch(
            "osf_sync.augmentation.doi_multi_method.process_reference", side_effect=_matched_row
        ):
            out = enrich_missing_with_multi_method(limit=25, osf_id="p9", workers=1)

        self.assertNotIn("preprints_selected", out)
        self.assertFalse(out["limit_reached"])
        self.assertEqual(out["checked"], 1)
        self.assertEqual(repo.preprint_selector_calls, 0)
        self.assertEqual(repo.select_refs_calls[0]["osf_id"], "p9")
        self.assertEqual(repo.select_refs_calls[0]["limit"], 25)


if __name__ == "__main__":
    unittest.main()
