import unittest

from osf_sync.extraction.get_mail_from_pdf import (
    _extract_emails_from_text,
    _normalize_pdf_text_for_email_extraction,
)


class PdfEmailExtractionTests(unittest.TestCase):
    def test_recovers_email_with_bracketed_at_dot_tokens(self) -> None:
        text = "Contact: jane [at] uni [dot] edu"
        normalized = _normalize_pdf_text_for_email_extraction(text)
        emails = _extract_emails_from_text(normalized)
        self.assertIn("jane@uni.edu", [e.lower() for e in emails])

    def test_recovers_email_with_parenthesized_tokens(self) -> None:
        text = "Reach me at john (at) ed (dot) ac (dot) uk"
        normalized = _normalize_pdf_text_for_email_extraction(text)
        emails = _extract_emails_from_text(normalized)
        self.assertIn("john@ed.ac.uk", [e.lower() for e in emails])

    def test_recovers_email_with_spaces_around_symbols(self) -> None:
        text = "Contact: jane . doe @ uni . edu"
        normalized = _normalize_pdf_text_for_email_extraction(text)
        emails = _extract_emails_from_text(normalized)
        self.assertIn("jane.doe@uni.edu", [e.lower() for e in emails])

    def test_recovers_email_split_across_lines(self) -> None:
        text = "Reach me at\njohn.doe@ed.ac.\nuk for details"
        normalized = _normalize_pdf_text_for_email_extraction(text)
        emails = _extract_emails_from_text(normalized)
        self.assertIn("john.doe@ed.ac.uk", [e.lower() for e in emails])

    def test_discards_long_suffix_garbage_after_tld(self) -> None:
        text = "vasildinev@gmail.comBulgarianAcademyofSciences"
        normalized = _normalize_pdf_text_for_email_extraction(text)
        emails = _extract_emails_from_text(normalized)
        self.assertEqual(emails, [])

    def test_repairs_common_location_prefix_noise(self) -> None:
        text = "Berlin.cornelius.erfort@hu-berlin.de"
        normalized = _normalize_pdf_text_for_email_extraction(text)
        emails = _extract_emails_from_text(normalized)
        self.assertIn("cornelius.erfort@hu-berlin.de", [e.lower() for e in emails])


if __name__ == "__main__":
    unittest.main()
