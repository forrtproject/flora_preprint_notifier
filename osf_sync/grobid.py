from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Tuple

from requests.exceptions import RequestException, Timeout, ConnectionError
from .dynamo.preprints_repo import PreprintsRepo
from .iter_preprints import SESSION
from .tei_cache import upload_tei

GROBID_URL = os.environ.get("GROBID_URL")
DATA_ROOT   = os.environ.get("PDF_DEST_ROOT", "/data/preprints")
INCLUDE_RAW_CITATIONS = os.environ.get("GROBID_INCLUDE_RAW_CITATIONS", "true").lower() in {"1", "true", "yes"}

def _pdf_path(provider_id: str, osf_id: str) -> Optional[Path]:
    # New structure
    p = Path(DATA_ROOT) / provider_id / osf_id / "file.pdf"
    if p.exists():
        return p
    # Legacy fallback (no provider)
    p_old = Path(DATA_ROOT) / osf_id / "file.pdf"
    if p_old.exists():
        # Optionally, move into new structure for cleanliness
        new_dir = Path(DATA_ROOT) / provider_id / osf_id
        new_dir.mkdir(parents=True, exist_ok=True)
        new_dst = new_dir / "file.pdf"
        try:
            p_old.replace(new_dst)
            return new_dst
        except Exception:
            return p_old
    return None

def _tei_output_path(provider_id: str, osf_id: str) -> Path:
    d = Path(DATA_ROOT) / provider_id / osf_id
    d.mkdir(parents=True, exist_ok=True)
    return d / "tei.xml"

def process_pdf_to_tei(provider_id: str, osf_id: str) -> Tuple[bool, Optional[str], Optional[str]]:
    pdf = _pdf_path(provider_id, osf_id)
    if not pdf:
        return (False, None, "PDF missing")
    if not GROBID_URL:
        return (False, None, "GROBID_URL is not set")

    url = f"{GROBID_URL.rstrip('/')}/api/processFulltextDocument"
    files = {"input": ("file.pdf", open(pdf, "rb"), "application/pdf")}
    params = {"consolidateHeader": "1"}
    if INCLUDE_RAW_CITATIONS:
        params["includeRawCitations"] = "1"
    try:
        with SESSION.post(url, files=files, data=params, timeout=(10, 120)) as r:
            r.raise_for_status()
            tei_path = _tei_output_path(provider_id, osf_id)
            tmp = tei_path.with_suffix(".xml.tmp")
            tmp.write_text(r.text, encoding="utf-8")
            tmp.replace(tei_path)
            upload_tei(provider_id, osf_id, str(tei_path))
            return (True, str(tei_path), None)
    except (RequestException, Timeout, ConnectionError) as e:
        return (False, None, str(e))

def mark_tei(osf_id: str, ok: bool, tei_path: Optional[str]):
    repo = PreprintsRepo()
    repo.mark_tei(osf_id, ok=ok, tei_path=tei_path)
