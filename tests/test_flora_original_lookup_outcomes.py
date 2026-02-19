import unittest

from osf_sync.augmentation.flora_original_lookup import _extract_ref_objects


class FloraOutcomeExtractionTests(unittest.TestCase):
    def test_extracts_replication_outcome_from_nested_replications(self) -> None:
        payload = {
            "results": {
                "10.1234/example": {
                    "originals": [{"doi": "10.1234/EXAMPLE", "apa_ref": "Original APA"}],
                    "replications": [
                        {"doi": "10.9999/rep1", "apa_ref": "Rep APA 1", "outcome": "success"},
                        {"doi": "10.9999/rep2", "apa_ref": "Rep APA 2", "outcome": "failed"},
                    ],
                }
            }
        }

        out = _extract_ref_objects(payload)

        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["doi_o"], "10.1234/example")
        self.assertEqual(out[0]["doi_r"], "10.9999/rep1")
        self.assertEqual(out[0]["replication_outcome"], "successful")
        self.assertEqual(out[1]["doi_r"], "10.9999/rep2")
        self.assertEqual(out[1]["replication_outcome"], "failed")

    def test_extracts_replication_outcome_from_flat_shapes(self) -> None:
        payload = [
            {"doi_o": "10.1234/example", "doi_r": "10.9999/rep1", "outcome_r": "mixed"},
            {"doi_o": "10.1234/example", "doi_r": "10.9999/rep2", "replication_outcome": "success"},
        ]

        out = _extract_ref_objects(payload)

        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["replication_outcome"], "mixed")
        self.assertEqual(out[1]["replication_outcome"], "successful")

    def test_dedup_prefers_non_empty_outcome(self) -> None:
        payload = [
            {"doi_o": "10.1234/example", "doi_r": "10.9999/rep1", "apa_ref_o": "O", "apa_ref_r": "R"},
            {
                "doi_o": "10.1234/example",
                "doi_r": "10.9999/rep1",
                "apa_ref_o": "O",
                "apa_ref_r": "R",
                "outcome": "failed",
            },
        ]

        out = _extract_ref_objects(payload)

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["replication_outcome"], "failed")

    def test_missing_outcome_is_retained_for_legacy_payloads(self) -> None:
        payload = [
            {"doi_o": "10.1234/example", "doi_r": "10.9999/rep1", "apa_ref_o": "O", "apa_ref_r": "R"},
        ]

        out = _extract_ref_objects(payload)

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["replication_outcome"], "unknown")


if __name__ == "__main__":
    unittest.main()
