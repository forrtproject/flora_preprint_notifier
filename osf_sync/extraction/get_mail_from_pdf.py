import re
from pypdf import PdfReader

# Function that attempts to read a text from a PDF and find all email adresses. 
# Input is a path to a specific PDF file, output is a list of all emails found in a specific PDF
# (returns an empty list if no emails are found)

# Logs saying "ignoring wrong pointing object x y (offset 0) are a byproduct of using pypdf and can be safely ignored"

STRICT_EMAIL_RE = re.compile(
    pattern=r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,15}\b",
    flags=re.IGNORECASE | re.MULTILINE | re.UNICODE,
)
LOOSE_EMAIL_RE = re.compile(
    pattern=(
        r"\b[A-Za-z0-9._%+\-]+(?:\s+[A-Za-z0-9._%+\-]+){0,4}\s*@\s*"
        r"[A-Za-z0-9.\-]+(?:\s+[A-Za-z0-9.\-]+){0,4}\s*\.\s*[A-Za-z]{2,15}\b"
    ),
    flags=re.IGNORECASE | re.MULTILINE | re.UNICODE,
)


def _normalize_pdf_text_for_email_extraction(text: str) -> str:
    if not text:
        return ""
    # Remove invisible separators and rejoin hyphenated line wraps.
    txt = text.replace("\u00ad", "").replace("\u200b", "")
    txt = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", txt)
    # Repair common spacing artifacts around email separators.
    txt = re.sub(r"\s*@\s*", "@", txt)
    txt = re.sub(r"\s*\.\s*", ".", txt)
    txt = re.sub(r"\s*_\s*", "_", txt)
    txt = re.sub(r"\s*\+\s*", "+", txt)
    txt = re.sub(r"\s*-\s*", "-", txt)
    return txt


def _repair_common_prefix_noise(email: str) -> str:
    if "@" not in email:
        return email
    local, domain = email.split("@", 1)
    # Example artifact: Berlin.cornelius.erfort@...
    dot_parts = local.split(".")
    if len(dot_parts) >= 3 and dot_parts[0][:1].isupper() and dot_parts[0][1:].islower():
        candidate = ".".join(dot_parts[1:]) + "@" + domain
        if STRICT_EMAIL_RE.fullmatch(candidate):
            return candidate
    # Example artifact: SingaporeNikita_Rane@...
    match = re.match(r"^([A-Z][a-z]{3,})([A-Z][A-Za-z0-9._%+\-]+)$", local)
    if match and any(ch in match.group(2) for ch in "._-"):
        candidate = match.group(2) + "@" + domain
        if STRICT_EMAIL_RE.fullmatch(candidate):
            return candidate
    return email


def _extract_emails_from_text(text: str) -> list[str]:
    if not text:
        return []
    out = []
    for match in STRICT_EMAIL_RE.findall(text):
        out.append(match)
    # Fallback: allow embedded whitespace and repair inside-match spaces.
    for raw in LOOSE_EMAIL_RE.findall(text):
        compact = re.sub(r"\s+", "", raw)
        if STRICT_EMAIL_RE.fullmatch(compact):
            out.append(compact)
    # Stable de-duplication while preserving original case in first occurrence.
    seen = set()
    unique = []
    for item in out:
        repaired = _repair_common_prefix_noise(item)
        key = repaired.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(repaired)
    return unique


def get_mail_from_pdf(path):
    doc = PdfReader(path)
    article_emails = []

    for page in doc.pages:
        page_text = page.extract_text() or ""
        normalized_text = _normalize_pdf_text_for_email_extraction(page_text)
        page_matches = _extract_emails_from_text(normalized_text)
        article_emails.extend(page_matches)

    return article_emails
