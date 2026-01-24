from __future__ import annotations
from typing import Dict, List, Optional, Any
from ..dynamo.preprints_repo import PreprintsRepo
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

    repo = PreprintsRepo()

    try:
        # TEI summary upsert
        try:
            repo.upsert_tei(osf_id, {
                "title": p_title,
                "doi": p_doi,
                "authors": p_authors,
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

        # References
        for idx, ref in enumerate(references):
            ref_id = ref.get("ref_id") or f"r{idx}"
            item = {
                "ref_id": ref_id,
                "title": _safe_str(ref.get("title")),
                "authors": _ensure_list(ref.get("authors")),
                "journal": _safe_str(ref.get("journal")),
                "year": _to_int_or_none(ref.get("year")),
                "doi": _safe_str(ref.get("doi")),
                "has_doi": bool(ref.get("has_doi")),
                "has_title": bool(ref.get("has_title")),
                "has_authors": bool(ref.get("has_authors")),
                "has_journal": bool(ref.get("has_journal")),
                "has_year": bool(ref.get("has_year")),
                "doi_source": "tei" if ref.get("doi") else None,
                "raw_citation": _safe_str(ref.get("raw_citation")),
            }
            try:
                repo.upsert_reference(osf_id, item)
                result["refs_upserted"] += 1
            except Exception as e:
                result["refs_failed"] += 1
                _log.warning(
                    "Reference upsert failed",
                    extra={"osf_id": osf_id, "ref_idx": idx, "ref_id": ref_id, "error": str(e)},
                )
                if raise_on_error:
                    raise

        if result["tei_ok"]:
            repo.mark_extracted(osf_id)
            _log.info("Marked preprint as extracted", extra={"osf_id": osf_id})

        _log.info("TEI extraction write complete", extra=result)
        return result

    except Exception:
        _log.exception("Unexpected error during write_extraction", extra={"osf_id": osf_id})
        if raise_on_error:
            raise
        result["tei_ok"] = False
        return result
