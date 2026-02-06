import unittest

from osf_sync.extraction.extract_author_list import _assign_pdf_emails, _score_group_email_matches


class AuthorEmailSelectionTests(unittest.TestCase):
    def test_tei_email_is_used_before_other_sources(self) -> None:
        rows = [
            {
                "name.given": "Alice",
                "name.surname": "Smith",
                "email": "alice@uni.edu",
                "email.source": "xml",
                "osf.name": "Alice Smith",
                "n": 1,
            },
            {
                "name.given": "Bob",
                "name.surname": "Jones",
                "email": "bob@uni.edu",
                "email.source": "orcid",
                "osf.name": "Bob Jones",
                "n": 2,
            },
        ]

        candidates = _score_group_email_matches(rows, threshold=0.90)

        self.assertEqual(candidates, [{"name": "Alice Smith", "email": "alice@uni.edu"}])

    def test_pdf_email_assignment_requires_plausible_name_match(self) -> None:
        rows = [
            {
                "name.given": "Jane",
                "name.surname": "Doe",
                "email": None,
                "email.source": None,
                "osf.name": "Jane Doe",
                "n": 1,
            },
            {
                "name.given": "John",
                "name.surname": "Smith",
                "email": None,
                "email.source": None,
                "osf.name": "John Smith",
                "n": 2,
            },
        ]

        assigned = _assign_pdf_emails(
            rows,
            ["jane.doe@lab.edu", "contact@journal.org"],
            threshold=0.75,
        )

        self.assertEqual(assigned, 1)
        self.assertEqual(rows[0]["email"], "jane.doe@lab.edu")
        self.assertEqual(rows[0]["email.source"], "pdf")
        self.assertFalse(rows[1].get("email"))

        candidates = _score_group_email_matches(rows, threshold=0.90)
        self.assertEqual(candidates, [{"name": "Jane Doe", "email": "jane.doe@lab.edu"}])

    def test_orcid_fallback_prefers_first_and_last_author(self) -> None:
        rows = [
            {
                "name.given": "First",
                "name.surname": "Author",
                "email": "first@uni.edu",
                "email.source": "orcid",
                "osf.name": "First Author",
                "n": 1,
            },
            {
                "name.given": "Middle",
                "name.surname": "Author",
                "email": "middle@uni.edu",
                "email.source": "orcid",
                "osf.name": "Middle Author",
                "n": 2,
            },
            {
                "name.given": "Last",
                "name.surname": "Author",
                "email": "last@uni.edu",
                "email.source": "orcid",
                "osf.name": "Last Author",
                "n": 3,
            },
        ]

        candidates = _score_group_email_matches(rows, threshold=0.90)

        self.assertEqual(
            candidates,
            [
                {"name": "First Author", "email": "first@uni.edu"},
                {"name": "Last Author", "email": "last@uni.edu"},
            ],
        )

    def test_orcid_fallback_uses_first_up_to_three_when_last_missing(self) -> None:
        rows = [
            {
                "name.given": "First",
                "name.surname": "Author",
                "email": "first@uni.edu",
                "email.source": "orcid",
                "osf.name": "First Author",
                "n": 1,
            },
            {
                "name.given": "Second",
                "name.surname": "Author",
                "email": "second@uni.edu",
                "email.source": "orcid",
                "osf.name": "Second Author",
                "n": 2,
            },
            {
                "name.given": "Third",
                "name.surname": "Author",
                "email": "third@uni.edu",
                "email.source": "orcid",
                "osf.name": "Third Author",
                "n": 3,
            },
            {
                "name.given": "Fourth",
                "name.surname": "Author",
                "email": None,
                "email.source": None,
                "osf.name": "Fourth Author",
                "n": 4,
            },
        ]

        candidates = _score_group_email_matches(rows, threshold=0.90)

        self.assertEqual(
            candidates,
            [
                {"name": "First Author", "email": "first@uni.edu"},
                {"name": "Second Author", "email": "second@uni.edu"},
                {"name": "Third Author", "email": "third@uni.edu"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
