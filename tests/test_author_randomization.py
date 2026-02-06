import unittest

from osf_sync.author_randomization import (
    ComponentSummary,
    assign_components_balanced,
    compute_large_author_threshold,
    resolve_author_nodes_from_tokens,
    select_author_positions,
)


class AuthorRandomizationTests(unittest.TestCase):
    def test_large_author_threshold_95th_percentile(self) -> None:
        counts = [1, 2, 3, 4, 20]
        self.assertEqual(compute_large_author_threshold(counts, percentile=0.95), 20)

    def test_select_author_positions_first_two_last_over_threshold(self) -> None:
        self.assertEqual(select_author_positions(8, 4), [0, 1, 7])
        self.assertEqual(select_author_positions(3, 4), [0, 1, 2])

    def test_resolve_nodes_aggressive_collapse(self) -> None:
        token_groups = [
            ["orcid:0000-0000-0000-0001", "nameinit:smith|j"],
            ["osf:abc123", "namefull:smith|john", "nameinit:smith|j"],
            ["namefull:smith|jane", "nameinit:smith|j"],
            ["orcid:0000-0000-0000-0002", "nameinit:smith|m"],
        ]
        node_ids = resolve_author_nodes_from_tokens(token_groups)

        self.assertEqual(node_ids[0], node_ids[1])
        self.assertEqual(node_ids[1], node_ids[2])
        self.assertNotEqual(node_ids[0], node_ids[3])

    def test_assign_balances_contactable_preprints(self) -> None:
        components = {
            "C000001": ComponentSummary(
                cluster_id="C000001",
                node_ids=["N1"],
                preprint_ids=["p1"],
                stratum="psyarxiv",
                contactable_preprints=1,
                contactable_emails=9,
            ),
            "C000002": ComponentSummary(
                cluster_id="C000002",
                node_ids=["N2"],
                preprint_ids=["p2"],
                stratum="psyarxiv",
                contactable_preprints=1,
                contactable_emails=2,
            ),
            "C000003": ComponentSummary(
                cluster_id="C000003",
                node_ids=["N3"],
                preprint_ids=["p3"],
                stratum="psyarxiv",
                contactable_preprints=1,
                contactable_emails=1,
            ),
            "C000004": ComponentSummary(
                cluster_id="C000004",
                node_ids=["N4"],
                preprint_ids=["p4"],
                stratum="psyarxiv",
                contactable_preprints=1,
                contactable_emails=1,
            ),
        }

        assignments = assign_components_balanced(components, seed=17)

        treatment_preprints = 0
        control_preprints = 0
        for cluster_id, arm in assignments.items():
            if arm == "treatment":
                treatment_preprints += components[cluster_id].contactable_preprints
            else:
                control_preprints += components[cluster_id].contactable_preprints

        self.assertLessEqual(abs(treatment_preprints - control_preprints), 1)


if __name__ == "__main__":
    unittest.main()
