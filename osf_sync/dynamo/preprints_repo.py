from .client import get_dynamo_resource
from ..logging_setup import get_logger, with_extras
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key
from typing import List, Dict, Optional, Any
import datetime as dt


def _strip_nones(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of dict without None values (DynamoDB rejects None)."""
    return {k: v for k, v in d.items() if v is not None}

log = get_logger(__name__)

class PreprintsRepo:
    def __init__(self):
        ddb = get_dynamo_resource()
        self.t_preprints = ddb.Table("preprints")
        self.t_refs = ddb.Table("preprint_references")
        self.t_tei = ddb.Table("preprint_tei")
        self.t_sync = ddb.Table("sync_state")

    # --- cursors (sync_state) ---
    def get_cursor(self, source_key: str) -> Optional[str]:
        r = self.t_sync.get_item(Key={"source_key": source_key}).get("Item")
        return r.get("last_seen_published") if r else None

    def set_cursor(self, source_key: str, last_seen_iso: str) -> None:
        now = dt.datetime.utcnow().isoformat()
        self.t_sync.put_item(Item={
            "source_key": source_key,
            "last_seen_published": last_seen_iso,
            "last_run_at": now
        })

    # --- upsert preprints batch ---
    def upsert_preprints(self, rows: List[Dict]) -> int:
        # Dynamo BatchWrite (25/item max per request); keep it simple
        count = 0
        with self.t_preprints.batch_writer(overwrite_by_pkeys=["osf_id"]) as bw:
            for obj in rows:
                try:
                    a = (obj.get("attributes") or {})
                    rid = obj.get("relationships") or {}
                    # compute simple presence of a primary_file to avoid scanning nested JSON later
                    rel_pf = (rid.get("primary_file") or {})
                    has_primary = bool((rel_pf.get("data") or rel_pf.get("links")))
                    is_published = bool(a.get("is_published"))

                    item_full = {
                        "osf_id": obj["id"],
                        "type": obj.get("type"),
                        "provider_id": (rid.get("provider") or {}).get("data", {}).get("id"),
                        "title": a.get("title"),
                        "description": a.get("description"),
                        "doi": a.get("doi"),
                        "date_created": a.get("date_created"),
                        "date_modified": a.get("date_modified"),
                        "date_published": a.get("date_published") or "",
                        "is_published": is_published,
                        "version": a.get("version"),
                        "is_latest_version": a.get("is_latest_version"),
                        "reviews_state": a.get("reviews_state"),
                        "tags": a.get("tags") or [],
                        "subjects": a.get("subjects") or [],
                        "license_record": a.get("license_record"),
                        "links": obj.get("links"),
                        "raw": obj,
                        "pdf_downloaded": False,
                        "tei_generated": False,
                        "tei_extracted": False,
                        "updated_at": dt.datetime.utcnow().isoformat()
                    }
                    # queue flags for GSIs
                    if is_published and has_primary:
                        item_full["queue_pdf"] = "pending"
                    bw.put_item(Item=_strip_nones(item_full))
                    count += 1
                except Exception as e:
                    with_extras(log, osf_id=obj.get("id"), err=str(e)).warning("preprint upsert failed")
        return count

    # --- PDF / TEI flags ---
    def mark_pdf(self, osf_id: str, ok: bool, path: Optional[str] = None):
        now = dt.datetime.utcnow().isoformat()
        if ok and path:
            self.t_preprints.update_item(
                Key={"osf_id": osf_id},
                UpdateExpression=(
                    "SET pdf_downloaded=:ok, pdf_path=:p, pdf_downloaded_at=:t, updated_at=:t, "
                    "queue_pdf=:done, queue_grobid=:pending"
                ),
                ExpressionAttributeValues={":ok": True, ":p": path, ":t": now, ":done": "done", ":pending": "pending"},
            )
        else:
            # Only set flag/timestamp, avoid writing None path
            self.t_preprints.update_item(
                Key={"osf_id": osf_id},
                UpdateExpression=(
                    "SET pdf_downloaded=:ok, pdf_downloaded_at=:t, updated_at=:t, queue_pdf=:done"
                ),
                ExpressionAttributeValues={":ok": bool(ok), ":t": now, ":done": "done"},
            )

    def mark_tei(self, osf_id: str, ok: bool, tei_path: Optional[str]):
        now = dt.datetime.utcnow().isoformat()
        if ok and tei_path:
            self.t_preprints.update_item(
                Key={"osf_id": osf_id},
                UpdateExpression=(
                    "SET tei_generated=:ok, tei_path=:p, tei_generated_at=:t, updated_at=:t, "
                    "queue_grobid=:done, queue_extract=:pending"
                ),
                ExpressionAttributeValues={":ok": True, ":p": tei_path, ":t": now, ":done": "done", ":pending": "pending"},
            )
        else:
            self.t_preprints.update_item(
                Key={"osf_id": osf_id},
                UpdateExpression=(
                    "SET tei_generated=:ok, tei_generated_at=:t, updated_at=:t, queue_grobid=:done"
                ),
                ExpressionAttributeValues={":ok": bool(ok), ":t": now, ":done": "done"},
            )

    def mark_extracted(self, osf_id: str):
        self.t_preprints.update_item(
            Key={"osf_id": osf_id},
            UpdateExpression=(
                "SET tei_extracted=:v, updated_at=:t, queue_extract=:done"
            ),
            ExpressionAttributeValues={":v": True, ":t": dt.datetime.utcnow().isoformat(), ":done": "done"}
        )

    # --- select queues ---
    def select_for_pdf(self, limit: int) -> List[str]:
        # Prefer GSI, fallback to scan
        try:
            resp = self.t_preprints.query(
                IndexName="by_queue_pdf",
                KeyConditionExpression="queue_pdf = :q",
                ExpressionAttributeValues={":q": "pending"},
                Limit=limit,
                ScanIndexForward=True,
            )
            return [it["osf_id"] for it in resp.get("Items", [])]
        except Exception:
            resp = self.t_preprints.scan(
                FilterExpression="is_published = :true AND (pdf_downloaded = :false OR attribute_not_exists(pdf_downloaded))",
                ExpressionAttributeValues={":true": True, ":false": False},
                Limit=limit,
            )
            return [it["osf_id"] for it in resp.get("Items", [])]

    def select_for_grobid(self, limit: int) -> List[str]:
        try:
            resp = self.t_preprints.query(
                IndexName="by_queue_grobid",
                KeyConditionExpression="queue_grobid = :q",
                ExpressionAttributeValues={":q": "pending"},
                Limit=limit,
                ScanIndexForward=True,
            )
            return [it["osf_id"] for it in resp.get("Items", [])]
        except Exception:
            resp = self.t_preprints.scan(
                FilterExpression="pdf_downloaded = :true AND (tei_generated = :false OR attribute_not_exists(tei_generated))",
                ExpressionAttributeValues={":true": True, ":false": False},
                Limit=limit,
            )
            return [it["osf_id"] for it in resp.get("Items", [])]

    def select_for_extraction(self, limit: int) -> List[Dict[str, Any]]:
        try:
            resp = self.t_preprints.query(
                IndexName="by_queue_extract",
                KeyConditionExpression="queue_extract = :q",
                ExpressionAttributeValues={":q": "pending"},
                Limit=limit,
                ScanIndexForward=True,
            )
            return resp.get("Items", [])
        except Exception:
            resp = self.t_preprints.scan(
                FilterExpression=(
                    "pdf_downloaded = :t AND tei_generated = :t AND (attribute_not_exists(tei_extracted) OR tei_extracted = :f)"
                ),
                ExpressionAttributeValues={":t": True, ":f": False},
                Limit=limit,
            )
            return resp.get("Items", [])

    # --- TEI / references writes ---
    def upsert_tei(self, osf_id: str, preprint: Dict) -> None:
        item_full = {"osf_id": osf_id, **preprint, "extracted_at": dt.datetime.utcnow().isoformat()}
        self.t_tei.put_item(Item=_strip_nones(item_full))

    def upsert_reference(self, osf_id: str, ref: Dict) -> None:
        item_full = {"osf_id": osf_id, "ref_id": ref["ref_id"], **ref,
                     "updated_at": dt.datetime.utcnow().isoformat()}
        self.t_refs.put_item(Item=_strip_nones(item_full))

    # --- utilities / other operations ---
    def delete_preprint(self, osf_id: str) -> None:
        self.t_preprints.delete_item(Key={"osf_id": osf_id})

    def exists_preprint(self, osf_id: str) -> bool:
        return bool(self.t_preprints.get_item(Key={"osf_id": osf_id}).get("Item"))

    def get_preprint_basic(self, osf_id: str) -> Optional[Dict[str, Any]]:
        it = self.t_preprints.get_item(Key={"osf_id": osf_id}).get("Item")
        if not it:
            return None
        return {"osf_id": it.get("osf_id"), "provider_id": it.get("provider_id"), "raw": it.get("raw")}

    def get_preprint_doi(self, osf_id: str) -> Optional[str]:
        """
        Return the DOI stored on the preprint record (if any).
        """
        it = self.t_preprints.get_item(Key={"osf_id": osf_id}).get("Item")
        if not it:
            return None
        return it.get("doi")

    def select_refs_missing_doi(
        self,
        limit: int,
        osf_id: Optional[str] = None,
        *,
        ref_id: Optional[str] = None,
        include_existing: bool = False,
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []

        if osf_id:
            last_key = None
            while True:
                kwargs = {"KeyConditionExpression": Key("osf_id").eq(osf_id)}
                if last_key:
                    kwargs["ExclusiveStartKey"] = last_key
                resp = self.t_refs.query(**kwargs)
                chunk = resp.get("Items", [])
                items.extend(chunk)
                last_key = resp.get("LastEvaluatedKey")
                if not last_key or (limit and len(items) >= limit):
                    break
        else:
            fe = "(attribute_not_exists(doi) OR doi = :empty)"
            eav = {":empty": ""}
            resp = self.t_refs.scan(FilterExpression=fe, ExpressionAttributeValues=eav, Limit=limit)
            items = resp.get("Items", [])

        if ref_id:
            items = [it for it in items if it and it.get("ref_id") == ref_id]
            if include_existing and not items and osf_id:
                item = self.t_refs.get_item(Key={"osf_id": osf_id, "ref_id": ref_id}).get("Item")
                items = [item] if item else []

        if not include_existing:
            filtered = []
            for it in items:
                if not it:
                    continue
                doi_val = (it.get("doi") or "").strip()
                if not doi_val:
                    filtered.append(it)
            items = filtered

        if limit:
            items = items[:limit]
        return items

    def select_refs_with_doi(
        self,
        limit: int,
        osf_id: Optional[str] = None,
        *,
        ref_id: Optional[str] = None,
        only_unchecked: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Return references that already have a DOI. Optionally restrict to rows without FORRT status.
        """
        items: List[Dict[str, Any]] = []

        if osf_id:
            last_key = None
            while True:
                kwargs = {"KeyConditionExpression": Key("osf_id").eq(osf_id)}
                if last_key:
                    kwargs["ExclusiveStartKey"] = last_key
                resp = self.t_refs.query(**kwargs)
                chunk = resp.get("Items", [])
                items.extend(chunk)
                last_key = resp.get("LastEvaluatedKey")
                if not last_key or (limit and len(items) >= limit):
                    break
        else:
            fe = "(attribute_exists(doi) AND doi <> :empty)"
            eav = {":empty": ""}
            resp = self.t_refs.scan(FilterExpression=fe, ExpressionAttributeValues=eav, Limit=limit)
            items = resp.get("Items", [])

        if ref_id:
            items = [it for it in items if it and it.get("ref_id") == ref_id]

        if only_unchecked:
            filtered: List[Dict[str, Any]] = []
            for it in items:
                if not it:
                    continue
                status_val = it.get("forrt_lookup_status")
                # retry if no status yet, or explicitly False
                if status_val in (None, False):
                    filtered.append(it)
            items = filtered

        if limit:
            items = items[:limit]
        return items

    def update_reference_doi(self, osf_id: str, ref_id: str, doi: str, *, source: str) -> bool:
        # Only set if not already set
        now = dt.datetime.utcnow().isoformat()
        try:
            self.t_refs.update_item(
                Key={"osf_id": osf_id, "ref_id": ref_id},
                UpdateExpression="SET doi=:d, has_doi=:hd, doi_source=:src, updated_at=:t",
                ExpressionAttributeValues={":d": doi, ":hd": True, ":src": source, ":t": now, ":empty": ""},
                ConditionExpression="attribute_not_exists(doi) OR doi = :empty",
                ReturnValues="NONE",
            )
            return True
        except ClientError as e:
            # Conditional check failed or other issue
            if e.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                return False
            raise

    def update_reference_forrt(
        self,
        osf_id: str,
        ref_id: str,
        *,
        status: bool,
        payload: Optional[Dict[str, Any]] = None,
        ref_objects: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Persist FORRT lookup result onto a reference row.
        """
        now = dt.datetime.utcnow().isoformat()

        expr_parts = ["forrt_lookup_status=:s", "forrt_checked_at=:t"]
        eav: Dict[str, Any] = {":s": bool(status), ":t": now}

        if payload is not None:
            expr_parts.append("forrt_lookup_payload=:p")
            eav[":p"] = payload
        if ref_objects is not None:
            expr_parts.append("forrt_refs=:fr")
            eav[":fr"] = ref_objects

        ue = "SET " + ", ".join(expr_parts)

        self.t_refs.update_item(
            Key={"osf_id": osf_id, "ref_id": ref_id},
            UpdateExpression=ue,
            ExpressionAttributeValues=eav,
            ReturnValues="NONE",
        )

    def select_refs_with_forrt_original(
        self,
        limit: int,
        osf_id: Optional[str] = None,
        *,
        ref_id: Optional[str] = None,
        include_missing_original: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return references that have a FORRT lookup (status/output); original DOI field is no longer used.
        """
        items: List[Dict[str, Any]] = []

        if osf_id:
            last_key = None
            while True:
                kwargs = {"KeyConditionExpression": Key("osf_id").eq(osf_id)}
                if last_key:
                    kwargs["ExclusiveStartKey"] = last_key
                resp = self.t_refs.query(**kwargs)
                chunk = resp.get("Items", [])
                items.extend(chunk)
                last_key = resp.get("LastEvaluatedKey")
                if not last_key or (limit and len(items) >= limit):
                    break
        else:
            fe = "attribute_exists(forrt_lookup_status)"
            resp = self.t_refs.scan(FilterExpression=fe, Limit=limit)
            items = resp.get("Items", [])

        if ref_id:
            items = [it for it in items if it and it.get("ref_id") == ref_id]

        # Only keep rows that actually carry the original DOI attribute
        if not include_missing_original:
            filtered = []
            for it in items:
                if not it:
                    continue
                status_val = it.get("forrt_lookup_status")
                if status_val:
                    filtered.append(it)
            items = filtered

        if limit:
            items = items[:limit]
        return items

    def update_reference_forrt_screening(
        self,
        osf_id: str,
        ref_id: str,
        *,
        original_cited: bool,
    ) -> None:
        """
        Persist whether the FORRT original DOI is already cited in the same reference list.
        """
        now = dt.datetime.utcnow().isoformat()
        self.t_refs.update_item(
            Key={"osf_id": osf_id, "ref_id": ref_id},
            UpdateExpression="SET forrt_original_cited=:c, forrt_screened_at=:t",
            ExpressionAttributeValues={":c": bool(original_cited), ":t": now},
            ReturnValues="NONE",
        )
