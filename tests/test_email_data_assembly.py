import unittest

from osf_sync.email.data_assembly import _build_original_entry


class EmailDataAssemblyTests(unittest.TestCase):
    def test_build_original_entry_uses_csv_flora_keys(self) -> None:
        ref = {
            "raw_citation": "Original citation",
            "doi": "10.1000/original",
        }
        ref_pairs = [
            {
                "doi_o": "10.1000/original",
                "doi_r": "10.2000/replication",
                "apa_ref_o": "Original APA",
                "apa_ref_r": "Replication APA",
                "replication_outcome": "failed",
            }
        ]

        out = _build_original_entry(ref, ref_pairs)

        self.assertIsNotNone(out)
        replications = out["replications"]
        self.assertEqual(len(replications), 1)
        self.assertEqual(replications[0]["doi"], "10.2000/replication")
        self.assertEqual(replications[0]["full_reference"], "Replication APA")
        self.assertEqual(replications[0]["outcome"], "failed")


if __name__ == "__main__":
    unittest.main()
