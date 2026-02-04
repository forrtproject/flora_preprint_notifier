from .client import get_dynamo_resource
from ..logging_setup import get_logger, with_extras
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key
from typing import List, Dict, Optional, Any, Iterable, Set
import datetime as dt
import os


def _strip_nones(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of dict without None values (DynamoDB rejects None)."""
    return {k: v for k, v in d.items() if v is not None}

log = get_logger(__name__)

class PreprintsRepo:
    def __init__(self):
        ddb = get_dynamo_resource()
        self.ddb = ddb
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
        skip_existing = os.environ.get("OSF_INGEST_SKIP_EXISTING", "false").lower() in {"1", "true", "yes"}
        if skip_existing and rows:
            ids = [r.get("id") for r in rows if r.get("id")]
            existing = self._fetch_existing_ids(ids)
            if existing:
                before = len(rows)
                rows = [r for r in rows if r.get("id") not in existing]
                skipped = before - len(rows)
                if skipped:
                    with_extras(log, skipped=skipped, incoming=before).info("skipping existing preprints")
            if not rows:
                return 0
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

    def _fetch_existing_ids(self, ids: Iterable[str]) -> Set[str]:
        """
        Batch-get existing osf_id values to support skip-existing ingest.
        """
        key_schema = self.t_preprints.key_schema or []
        hash_key = next((k["AttributeName"] for k in key_schema if k.get("KeyType") == "HASH"), None)
        range_key = next((k["AttributeName"] for k in key_schema if k.get("KeyType") == "RANGE"), None)
        if not hash_key:
            with_extras(log, table=self.t_preprints.name).warning("preprints table missing hash key; skipping exists check")
            return set()
        if range_key:
            with_extras(log, table=self.t_preprints.name, range_key=range_key).warning(
                "preprints table has range key; skipping exists check"
            )
            return set()
        ddb = self.ddb
        table_name = self.t_preprints.name
        existing: Set[str] = set()
        id_list = [str(i) for i in ids if i]
        for chunk in _chunks(id_list, 100):
            keys = [{hash_key: i} for i in chunk]
            request = {table_name: {"Keys": keys, "ProjectionExpression": hash_key}}
            resp = ddb.batch_get_item(RequestItems=request)
            existing.update(_extract_key_values(resp.get("Responses", {}).get(table_name, []), hash_key))
            unprocessed = resp.get("UnprocessedKeys") or {}
            while unprocessed:
                resp = ddb.batch_get_item(RequestItems=unprocessed)
                existing.update(_extract_key_values(resp.get("Responses", {}).get(table_name, []), hash_key))
                unprocessed = resp.get("UnprocessedKeys") or {}
        return existing

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

    def select_refs_with_forrt_original(
        self,
        limit: int,
        osf_id: Optional[str] = None,
        *,
        ref_id: Optional[str] = None,
        include_missing_original: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return references that have been processed by FORRT lookup.
        If include_missing_original is True, include rows with status=False.
        """
        items: List[Dict[str, Any]] = []

        def _has_forrt(it: Dict[str, Any]) -> bool:
            status = it.get("forrt_lookup_status")
            has_payload = it.get("forrt_lookup_payload") is not None
            has_refs = bool(it.get("forrt_refs"))
            if include_missing_original:
                return (status is not None) or has_payload or has_refs
            return (status is True) or has_payload or has_refs

        if osf_id:
            last_key = None
            while True:
                kwargs = {"KeyConditionExpression": Key("osf_id").eq(osf_id)}
                if last_key:
                    kwargs["ExclusiveStartKey"] = last_key
                resp = self.t_refs.query(**kwargs)
                chunk = resp.get("Items", [])
                items.extend([it for it in chunk if it and _has_forrt(it)])
                last_key = resp.get("LastEvaluatedKey")
                if not last_key or (limit and len(items) >= limit):
                    break
        else:
            if include_missing_original:
                fe = "attribute_exists(forrt_lookup_status) OR attribute_exists(forrt_lookup_payload) OR attribute_exists(forrt_refs)"
                eav = None
            else:
                fe = "forrt_lookup_status = :true OR attribute_exists(forrt_lookup_payload) OR attribute_exists(forrt_refs)"
                eav = {":true": True}
            scan_kwargs = {"FilterExpression": fe, "Limit": limit}
            if eav is not None:
                scan_kwargs["ExpressionAttributeValues"] = eav
            resp = self.t_refs.scan(**scan_kwargs)
            items = resp.get("Items", [])

        if ref_id:
            items = [it for it in items if it and it.get("ref_id") == ref_id]
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

    def update_reference_raw_citation_validity(self, osf_id: str, ref_id: str, validity: str) -> None:
        now = dt.datetime.utcnow().isoformat()
        self.t_refs.update_item(
            Key={"osf_id": osf_id, "ref_id": ref_id},
            UpdateExpression="SET raw_citation_validity=:v, raw_citation_validity_updated_at=:t",
            ExpressionAttributeValues={":v": validity, ":t": now},
        )

    def update_reference_forrt(
        self,
        osf_id: str,
        ref_id: str,
        *,
        status: bool,
        ref_pairs: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        now = dt.datetime.utcnow().isoformat()
        set_exprs = ["forrt_lookup_status=:s", "forrt_checked_at=:t", "updated_at=:t"]
        remove_exprs = ["forrt_lookup_payload", "forrt_refs", "forrt_refs_count", "forrt_ref_pairs", "forrt_ref_pairs_count"]
        eav: Dict[str, Any] = {":s": bool(status), ":t": now}

        if ref_pairs is not None:
            set_exprs.append("forrt_ref_pairs=:p")
            eav[":p"] = ref_pairs
            set_exprs.append("forrt_ref_pairs_count=:pc")
            eav[":pc"] = len(ref_pairs)
            remove_exprs = [r for r in remove_exprs if r not in {"forrt_ref_pairs", "forrt_ref_pairs_count"}]

        update_expr = "SET " + ", ".join(set_exprs)
        if remove_exprs:
            update_expr += " REMOVE " + ", ".join(remove_exprs)

        self.t_refs.update_item(
            Key={"osf_id": osf_id, "ref_id": ref_id},
            UpdateExpression=update_expr,
            ExpressionAttributeValues=eav,
        )

    def update_reference_forrt_screening(
        self,
        osf_id: str,
        ref_id: str,
        *,
        original_cited: bool,
    ) -> None:
        now = dt.datetime.utcnow().isoformat()
        update_expr = "SET forrt_original_cited=:v, forrt_screened_at=:t, updated_at=:t REMOVE forrt_matching_replication_dois"
        self.t_refs.update_item(
            Key={"osf_id": osf_id, "ref_id": ref_id},
            UpdateExpression=update_expr,
            ExpressionAttributeValues={":v": bool(original_cited), ":t": now},
        )


def _chunks(seq: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _extract_key_values(items: List[Dict[str, Any]], key_name: str) -> Set[str]:
    out: Set[str] = set()
    for item in items:
        val = item.get(key_name)
        if isinstance(val, dict) and "S" in val:
            out.add(val.get("S"))
        elif isinstance(val, str):
            out.add(val)
    return out
