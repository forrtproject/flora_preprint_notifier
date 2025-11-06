from __future__ import annotations
from typing import Dict, List, Optional, Any
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from ..db import engine
import logging, traceback, json

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger.setLevel(logging.INFO)

def _json_dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)

def _ensure_list(v: Any): return v if isinstance(v, list) else ([] if v is None else [v])
def _safe_str(v: Any, maxlen: int | None = None): 
    if v is None: return None
    s = str(v);  return s if maxlen is None or len(s) <= maxlen else s[:maxlen]
def _to_int_or_none(v: Any):
    if v is None: return None
    s = str(v).strip();  return int(s) if s.isdigit() else None

UPSERT_TEI = text("""
INSERT INTO preprint_tei (
  osf_id, title, doi, authors, published_date,
  has_title, has_doi, has_authors, has_published_date, extracted_at
) VALUES (
  :osf_id, :title, :doi, :authors_json, :published_date,
  :has_title, :has_doi, :has_authors, :has_published_date, now()
)
ON CONFLICT (osf_id) DO UPDATE SET
  title = EXCLUDED.title,
  doi = EXCLUDED.doi,
  authors = EXCLUDED.authors,
  published_date = EXCLUDED.published_date,
  has_title = EXCLUDED.has_title,
  has_doi = EXCLUDED.has_doi,
  has_authors = EXCLUDED.has_authors,
  has_published_date = EXCLUDED.has_published_date,
  extracted_at = now()
""")

UPSERT_REF = text("""
INSERT INTO preprint_references (
  osf_id, ref_id, title, authors, journal, year,
  doi, has_doi, has_title, has_authors, has_journal, has_year,
  doi_source, updated_at
) VALUES (
  :osf_id, :ref_id, :title, :authors_json, :journal, :year,
  :doi, :has_doi, :has_title, :has_authors, :has_journal, :has_year,
  :doi_source, now()
)
ON CONFLICT (osf_id, ref_id) DO UPDATE SET
  title = EXCLUDED.title,
  authors = EXCLUDED.authors,
  journal = EXCLUDED.journal,
  year = EXCLUDED.year,
  doi = COALESCE(preprint_references.doi, EXCLUDED.doi),
  has_doi = EXCLUDED.has_doi OR preprint_references.has_doi,
  has_title = EXCLUDED.has_title,
  has_authors = EXCLUDED.has_authors,
  has_journal = EXCLUDED.has_journal,
  has_year = EXCLUDED.has_year,
  updated_at = now()
""")

MARK_EXTRACTED = text("""
UPDATE preprints
SET tei_extracted = TRUE, updated_at = now()
WHERE osf_id = :osf_id
""")

def write_extraction(
    osf_id: str,
    preprint: Dict,
    references: List[Dict],
    *,
    raise_on_error: bool = False,
    log: Optional[logging.Logger] = None,
) -> Dict[str, int | str | bool]:

    _log = log or logger
    _log.info("TEI upsert start", extra={"osf_id": osf_id, "refs": len(references)})

    result = {"osf_id": osf_id, "tei_ok": False, "refs_total": len(references), "refs_upserted": 0, "refs_failed": 0}

    p_title = _safe_str(preprint.get("title"))
    p_doi = _safe_str(preprint.get("doi"))
    p_authors = _ensure_list(preprint.get("authors"))
    p_published_date = _safe_str(preprint.get("published_date"))

    try:
        with engine.begin() as conn:
            # TEI upsert in SAVEPOINT
            try:
                with conn.begin_nested():
                    conn.execute(UPSERT_TEI, {
                        "osf_id": osf_id,
                        "title": p_title,
                        "doi": p_doi,
                        "authors_json": _json_dump(p_authors),
                        "published_date": p_published_date,
                        "has_title": bool(preprint.get("has_title")),
                        "has_doi": bool(preprint.get("has_doi")),
                        "has_authors": bool(preprint.get("has_authors")),
                        "has_published_date": bool(preprint.get("has_published_date")),
                    })
                result["tei_ok"] = True
                _log.info("TEI upserted", extra={"osf_id": osf_id, "title_snippet": (p_title or "")[:160]})
            except Exception as e:
                result["tei_ok"] = False
                _log.error("TEI upsert failed", extra={"osf_id": osf_id, "error": str(e)})
                if raise_on_error:
                    raise

            # Each reference in its own SAVEPOINT
            for idx, ref in enumerate(references):
                ref_id = ref.get("ref_id") or f"r{idx}"
                params = {
                    "osf_id": osf_id,
                    "ref_id": ref_id,
                    "title": _safe_str(ref.get("title")),
                    "authors_json": _json_dump(_ensure_list(ref.get("authors"))),
                    "journal": _safe_str(ref.get("journal")),
                    "year": _to_int_or_none(ref.get("year")),
                    "doi": _safe_str(ref.get("doi")),
                    "has_doi": bool(ref.get("has_doi")),
                    "has_title": bool(ref.get("has_title")),
                    "has_authors": bool(ref.get("has_authors")),
                    "has_journal": bool(ref.get("has_journal")),
                    "has_year": bool(ref.get("has_year")),
                    "doi_source": "tei" if ref.get("doi") else None,
                }
                try:
                    with conn.begin_nested():
                        conn.execute(UPSERT_REF, params)
                    result["refs_upserted"] += 1
                except Exception as e:
                    result["refs_failed"] += 1
                    _log.warning(
                        "Reference upsert failed",
                        extra={"osf_id": osf_id, "ref_idx": idx, "ref_id": ref_id, "error": str(e)},
                    )
                    if raise_on_error:
                        raise

            # Mark extracted only if TEI was ok
            if result["tei_ok"]:
                with conn.begin_nested():
                    conn.execute(MARK_EXTRACTED, {"osf_id": osf_id})
                _log.info("Marked preprint as extracted", extra={"osf_id": osf_id})

        _log.info("TEI extraction write complete", extra=result)
        return result

    except SQLAlchemyError:
        _log.exception("Database error during write_extraction", extra={"osf_id": osf_id})
        if raise_on_error:
            raise
        result["tei_ok"] = False
        return result
    except Exception:
        _log.exception("Unexpected error during write_extraction", extra={"osf_id": osf_id})
        if raise_on_error:
            raise
        result["tei_ok"] = False
        return result
