from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Any, Tuple

import jinja2
import markdown

logger = logging.getLogger(__name__)

_DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "email_draft.md"

_HTML_WRAPPER = """\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: sans-serif; line-height: 1.5; color: #333; max-width: 700px;">
{body}
</body>
</html>
"""


def load_template(path: Path | None = None) -> jinja2.Template:
    """Read the email markdown file and parse as a Jinja2 Template."""
    p = path or _DEFAULT_TEMPLATE_PATH
    text = p.read_text(encoding="utf-8")
    env = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False)
    return env.from_string(text)


def _md_to_plain(md: str) -> str:
    """Convert rendered markdown to plain text for the text/plain MIME part."""
    text = md
    # Convert markdown links [text](url) to "text (url)"
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    # Strip bold/italic markers
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    # Replace <br> / <br/> / <br /> with newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    # Strip remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def render_email(context: Dict[str, Any], *, template_path: Path | None = None) -> Tuple[str, str, str]:
    """Render the email template with the given context.

    Returns (subject, html_body, plain_text_body).
    """
    tmpl = load_template(template_path)
    rendered_md = tmpl.render(context)

    # Extract subject from first line: **Subject: ...**
    lines = rendered_md.split("\n", 1)
    subject_line = lines[0].strip()
    # Strip markdown bold markers and "Subject:" prefix
    subject = re.sub(r"^\*{1,2}", "", subject_line)
    subject = re.sub(r"\*{1,2}$", "", subject)
    subject = re.sub(r"^Subject:\s*", "", subject, flags=re.IGNORECASE).strip()

    # Body is everything after the first line
    body_md = lines[1] if len(lines) > 1 else ""

    # Convert markdown to HTML
    html_body = markdown.markdown(body_md.strip(), extensions=["extra"])

    # Plain text version
    plain_body = _md_to_plain(body_md.strip())

    return subject, _HTML_WRAPPER.format(body=html_body), plain_body
