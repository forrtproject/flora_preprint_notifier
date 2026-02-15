from __future__ import annotations
import os
import logging
import traceback
from .extract_to_db import write_extraction
from .extract_preprints_and_references import TEIExtractor

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
logger.setLevel(logging.INFO)


def extract_for_osf_id(provider_id: str, osf_id: str, base_dir: str, *, raise_on_error: bool = False) -> dict:
    """
    Locate TEI at: {base}/{provider}/{osf_id}/tei.xml,
    parse it with TEIExtractor, then write results to DB.

    If the TEI file is missing on disk (e.g. ephemeral GH Actions runner),
    attempts to regenerate it by re-downloading the PDF and running GROBID.

    Returns:
        {
          "osf_id": str,
          "tei_path": str,
          "parsed_ok": bool,
          "written_ok": bool,
          "refs_count": int,
          "error": Optional[str]
        }
    """
    tei_path = os.path.join(base_dir, provider_id, osf_id, "tei.xml")
    summary = {
        "osf_id": osf_id,
        "tei_path": tei_path,
        "parsed_ok": False,
        "written_ok": False,
        "refs_count": 0,
        "error": None,
    }

    try:
        # --- Check TEI existence, with ephemeral storage fallback ---
        if not os.path.exists(tei_path):
            logger.info(
                "TEI file missing on disk, attempting regeneration via GROBID",
                extra={"osf_id": osf_id, "provider": provider_id},
            )
            try:
                from ..pipeline import download_single_pdf
                from ..grobid import _pdf_path, process_pdf_to_tei

                if _pdf_path(provider_id, osf_id) is None:
                    download_single_pdf(osf_id)
                ok, generated_path, err = process_pdf_to_tei(provider_id, osf_id)
                if ok and generated_path:
                    tei_path = generated_path
                    summary["tei_path"] = tei_path
                    logger.info("TEI regenerated successfully", extra={"osf_id": osf_id})
                else:
                    msg = f"TEI regeneration failed: {err}"
                    logger.warning(msg, extra={"osf_id": osf_id, "provider": provider_id})
                    summary["error"] = msg
                    return summary
            except Exception as regen_err:
                msg = f"TEI regeneration error: {regen_err}"
                logger.warning(msg, extra={"osf_id": osf_id, "provider": provider_id})
                summary["error"] = msg
                return summary

        logger.info("Starting TEI parse", extra={"osf_id": osf_id, "provider": provider_id})

        # --- Parse TEI ---
        x = TEIExtractor()
        result = x.parse_file(tei_path)
        preprint = result.get("preprint") or {}
        refs = result.get("references") or []
        summary["refs_count"] = len(refs)
        summary["parsed_ok"] = True

        logger.info(
            "Parsed TEI successfully",
            extra={
                "osf_id": osf_id,
                "provider": provider_id,
                "refs_found": len(refs),
                "has_title": bool(preprint.get("title")),
            },
        )

        # --- Write to DB ---
        db_summary = write_extraction(osf_id, preprint, refs, raise_on_error=raise_on_error, log=logger)
        summary["written_ok"] = db_summary.get("tei_ok", False)
        summary["refs_count"] = db_summary.get("refs_upserted", summary["refs_count"])

        logger.info(
            "TEI extraction completed",
            extra={
                "osf_id": osf_id,
                "provider": provider_id,
                "written_ok": summary["written_ok"],
                "refs_upserted": summary["refs_count"],
            },
        )

    except Exception as e:
        tb = traceback.format_exc()
        summary["error"] = str(e)
        logger.error(
            "Error during TEI extraction",
            extra={
                "osf_id": osf_id,
                "provider": provider_id,
                "error": str(e),
                "traceback": tb,
            },
        )
        if raise_on_error:
            raise

    return summary
