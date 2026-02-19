import unittest

from osf_sync.augmentation.doi_multi_method_lookup import _score_candidate_parsed


class DoiMultiMethodAuthorOverlapTests(unittest.TestCase):
    def test_parsed_candidate_requires_author_overlap_when_both_sides_have_authors(self) -> None:
        cand = {
            "title": ["Replication and Memory"],
            "container-title": ["Journal of Testing"],
            "issued": {"date-parts": [[2024, 1, 1]]},
            "author": [{"given": "Alice", "family": "Smith"}],
        }

        score, meta = _score_candidate_parsed(
            ref_title="Replication and Memory",
            ref_journal="Journal of Testing",
            ref_year=2024,
            ref_authors=["Bob Jones"],
            cand=cand,
            method="crossref_title",
        )

        self.assertIsNone(score)
        self.assertEqual(meta.get("reason"), "author_mismatch")


if __name__ == "__main__":
    unittest.main()
