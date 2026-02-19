import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from osf_sync.augmentation import flora_original_lookup as fol


class _FakeRepo:
    def __init__(self, rows, *, unsent_ids=None):
        self.rows = rows
        self.updates = []
        self.unsent_ids = set(unsent_ids or [])

    def select_refs_with_doi(self, *, limit, osf_id, ref_id, only_unchecked):
        return self.rows[:limit] if limit else list(self.rows)

    def filter_osf_ids_without_sent_email(self, osf_ids):
        if not self.unsent_ids:
            return set(osf_ids)
        return {oid for oid in osf_ids if oid in self.unsent_ids}

    def update_reference_flora(self, osf_id, ref_id, *, status, ref_pairs=None):
        self.updates.append(
            {
                "osf_id": osf_id,
                "ref_id": ref_id,
                "status": status,
                "ref_pairs": ref_pairs or [],
            }
        )


class FloraOriginalLookupCsvTests(unittest.TestCase):
    def test_load_pairs_by_original_dedupes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "flora.csv"
            csv_path.write_text(
                "\ufeff\"doi_o\",\"doi_r\",\"apa_ref_o\",\"apa_ref_r\",\"outcome\"\n"
                "\"10.1000/abc\",\"10.2000/rep\",\"O1\",\"R1\",\"successful\"\n"
                "\"10.1000/abc\",\"10.2000/rep\",\"O1\",\"R1\",\"successful\"\n"
                "\"10.1000/abc\",\"\",\"O1\",\"\",\"mixed\"\n",
                encoding="utf-8",
            )

            pairs = fol._load_flora_pairs_by_original(csv_path)

            self.assertIn("10.1000/abc", pairs)
            self.assertEqual(len(pairs["10.1000/abc"]), 2)
            self.assertEqual(pairs["10.1000/abc"][0]["doi_r"], "10.2000/rep")
            self.assertIsNone(pairs["10.1000/abc"][1]["doi_r"])

    def test_lookup_uses_local_csv_and_marks_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "flora.csv"
            csv_path.write_text(
                "\ufeff\"doi_o\",\"doi_r\",\"apa_ref_o\",\"apa_ref_r\",\"outcome\"\n"
                "\"10.1000/abc\",\"10.2000/rep\",\"O1\",\"R1\",\"failed\"\n",
                encoding="utf-8",
            )
            repo = _FakeRepo(
                [
                    {"osf_id": "p1", "ref_id": "r1", "doi": "10.1000/abc"},
                    {"osf_id": "p1", "ref_id": "r2", "doi": "10.9999/missing"},
                ]
            )

            with patch("osf_sync.augmentation.flora_original_lookup.PreprintsRepo", return_value=repo):
                with patch("osf_sync.augmentation.flora_original_lookup._ensure_fresh_flora_csv", return_value={"downloaded": False, "used_stale": False}):
                    out = fol.lookup_originals_with_flora(limit=10, cache_path=str(csv_path))

            self.assertEqual(out["checked"], 2)
            self.assertEqual(out["updated"], 2)
            self.assertEqual(out["failed"], 0)
            self.assertEqual(len(repo.updates), 2)
            self.assertEqual(repo.updates[0]["status"], True)
            self.assertEqual(repo.updates[1]["status"], False)

    def test_lookup_skips_preprints_with_sent_email(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "flora.csv"
            csv_path.write_text(
                "\ufeff\"doi_o\",\"doi_r\",\"apa_ref_o\",\"apa_ref_r\",\"outcome\"\n"
                "\"10.1000/abc\",\"10.2000/rep\",\"O1\",\"R1\",\"mixed\"\n",
                encoding="utf-8",
            )
            repo = _FakeRepo(
                [
                    {"osf_id": "p_unsent", "ref_id": "r1", "doi": "10.1000/abc"},
                    {"osf_id": "p_sent", "ref_id": "r2", "doi": "10.1000/abc"},
                ],
                unsent_ids={"p_unsent"},
            )

            with patch("osf_sync.augmentation.flora_original_lookup.PreprintsRepo", return_value=repo):
                with patch(
                    "osf_sync.augmentation.flora_original_lookup._ensure_fresh_flora_csv",
                    return_value={"downloaded": False, "used_stale": False},
                ):
                    out = fol.lookup_originals_with_flora(limit=10, cache_path=str(csv_path))

            self.assertEqual(out["checked"], 1)
            self.assertEqual(out["updated"], 1)
            self.assertEqual(out["skipped_sent_preprint"], 1)
            self.assertEqual(len(repo.updates), 1)
            self.assertEqual(repo.updates[0]["osf_id"], "p_unsent")

    def test_load_pairs_filters_non_protocol_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "flora.csv"
            csv_path.write_text(
                "\ufeff\"doi_o\",\"doi_r\",\"apa_ref_o\",\"apa_ref_r\",\"outcome\"\n"
                "\"10.1000/abc\",\"10.2000/rep1\",\"O1\",\"R1\",\"descriptive only\"\n"
                "\"10.1000/abc\",\"10.2000/rep2\",\"O1\",\"R2\",\"successful\"\n",
                encoding="utf-8",
            )

            pairs = fol._load_flora_pairs_by_original(csv_path)

            self.assertIn("10.1000/abc", pairs)
            self.assertEqual(len(pairs["10.1000/abc"]), 1)
            self.assertEqual(pairs["10.1000/abc"][0]["doi_r"], "10.2000/rep2")
            self.assertEqual(pairs["10.1000/abc"][0]["replication_outcome"], "successful")


if __name__ == "__main__":
    unittest.main()
