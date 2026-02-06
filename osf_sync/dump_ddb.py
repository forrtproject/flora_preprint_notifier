#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import json
import csv
import argparse
from typing import Optional, List, Dict, Any
import boto3
from botocore.config import Config
from boto3.dynamodb.types import TypeDeserializer
from dotenv import load_dotenv

from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key

# Reuse app's Dynamo client (honors DYNAMO_LOCAL_URL or AWS creds)
from .dynamo.client import get_dynamo_resource

load_dotenv()


def _env_name(var: str, default: str) -> str:
    return os.getenv(var, default)


def _pretty(x) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2, default=str)


def _get_dynamo_client():
    local_url = os.getenv("DYNAMO_LOCAL_URL")
    region = os.getenv("AWS_REGION", "eu-central-1")
    cfg = Config(retries={"max_attempts": 10, "mode": "standard"})
    if local_url:
        return boto3.client(
            "dynamodb",
            region_name=region,
            endpoint_url=local_url,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "dummy"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "dummy"),
            config=cfg,
        )
    return boto3.client("dynamodb", region_name=region, config=cfg)


def list_tables(ddb):
    return [t.name for t in ddb.tables.all()]


def scan_some(ddb, table_name: str, limit: int):
    t = ddb.Table(table_name)
    try:
        resp = t.scan(Limit=limit)
        items = resp.get("Items", [])
        print(f"\n=== Table: {table_name} (showing {len(items)} items) ===")
        for it in items:
            print(_pretty(it))
    except ClientError as e:
        print(f"[ERROR] scan {table_name}: {e}")


def _get_nested(d: Dict[str, Any], keys: List[str]) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _build_raw_key_filter(raw_key: str, raw_path: str):
    path_parts = [p for p in raw_path.split(".") if p]
    expr_names: Dict[str, str] = {"#raw": "raw", "#key": raw_key}
    path_expr = "#raw"
    for i, part in enumerate(path_parts):
        expr_names[f"#p{i}"] = part
        path_expr += f".#p{i}"
    path_expr += ".#key"
    filter_expr = f"attribute_exists({path_expr})"
    return filter_expr, expr_names, path_parts


def scan_preprints_with_raw_key(
    ddb,
    raw_key: str,
    limit: int,
    raw_path: str = "attributes",
    *,
    writer=None,
    progress_every: int = 0,
    page_limit: Optional[int] = None,
    sleep_s: float = 0.0,
):
    t = ddb.Table(_env_name("DDB_TABLE_PREPRINTS", "preprints"))
    filter_expr, expr_names, path_parts = _build_raw_key_filter(raw_key, raw_path)

    last_key = None
    if page_limit is None:
        page_limit = max(50, min(1000, (limit or 100) * 5))
    label = f"raw.{raw_path}.{raw_key}" if raw_path else f"raw.{raw_key}"
    print(f"\n=== preprints with {label} ===")
    count = 0
    while True:
        kwargs: Dict[str, Any] = {
            "FilterExpression": filter_expr,
            "ExpressionAttributeNames": expr_names,
            "Limit": page_limit,
        }
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key
        resp = t.scan(**kwargs)
        for it in resp.get("Items", []):
            if limit and count >= limit:
                break
            raw = it.get("raw") or {}
            raw_value = _get_nested(raw, path_parts + [raw_key])
            if raw_value is None:
                continue
            row = {
                "osf_id": it.get("osf_id"),
                "provider_id": it.get("provider_id"),
                "date_published": it.get("date_published"),
                label: raw_value,
            }
            if writer:
                writer.write_row(row)
            else:
                print(_pretty(row))
            count += 1
            if progress_every and count % progress_every == 0:
                print(f"[progress] wrote {count} rows")
        if limit and count >= limit:
            break
        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break
        if sleep_s:
            time.sleep(sleep_s)
    return count


def _get_av_path(item_av: Dict[str, Any], path_parts: List[str]) -> Any:
    cur: Any = item_av
    for part in path_parts:
        if not isinstance(cur, dict):
            return None
        if "M" in cur and isinstance(cur.get("M"), dict):
            cur = cur.get("M")
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def scan_preprints_with_raw_key_av(
    raw_key: str,
    limit: int,
    raw_path: str = "attributes",
    *,
    writer=None,
    progress_every: int = 0,
    page_limit: Optional[int] = None,
    sleep_s: float = 0.0,
):
    client = _get_dynamo_client()
    table_name = _env_name("DDB_TABLE_PREPRINTS", "preprints")
    filter_expr, expr_names, path_parts = _build_raw_key_filter(raw_key, raw_path)
    dser = TypeDeserializer()

    last_key = None
    if page_limit is None:
        page_limit = max(50, min(1000, (limit or 100) * 5))
    label = f"raw.{raw_path}.{raw_key}" if raw_path else f"raw.{raw_key}"
    print(f"\n=== preprints with {label} (AV JSON) ===")
    count = 0
    backoff_s = 1.0
    while True:
        kwargs: Dict[str, Any] = {
            "TableName": table_name,
            "FilterExpression": filter_expr,
            "ExpressionAttributeNames": expr_names,
            "Limit": page_limit,
        }
        if last_key:
            kwargs["ExclusiveStartKey"] = last_key
        try:
            resp = client.scan(**kwargs)
            backoff_s = 1.0
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code == "ProvisionedThroughputExceededException":
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, 30.0)
                continue
            raise
        for it in resp.get("Items", []):
            if limit and count >= limit:
                break
            raw_av = _get_av_path(it, ["raw"] + path_parts + [raw_key])
            # Also show a deserialized scalar when possible.
            try:
                raw_val = dser.deserialize(raw_av) if raw_av is not None else None
            except Exception:
                raw_val = None
            if raw_val is None:
                continue
            row = {
                "osf_id": it.get("osf_id", {}).get("S"),
                "provider_id": it.get("provider_id", {}).get("S"),
                "date_published": it.get("date_published", {}).get("S"),
                label: raw_av,
                f"{label}_value": raw_val,
            }
            if writer:
                writer.write_row(row)
            else:
                print(_pretty(row))
            count += 1
            if progress_every and count % progress_every == 0:
                print(f"[progress] wrote {count} rows")
        if limit and count >= limit:
            break
        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break
        if sleep_s:
            time.sleep(sleep_s)
    return count


QUEUE_INDEXES = {
    "by_queue_pdf": ("queue_pdf", "date_published"),
    "by_queue_grobid": ("queue_grobid", "pdf_downloaded_at"),
    "by_queue_extract": ("queue_extract", "date_published"),
}


def query_queue(ddb, index_name: str, limit: int):
    t = ddb.Table(_env_name("DDB_TABLE_PREPRINTS", "preprints"))
    print(f"\n=== GSI: {index_name} (pending, limit {limit}) ===")
    attrs = QUEUE_INDEXES.get(index_name)
    if not attrs:
        print(f"[WARN] Unknown index mapping for {index_name}")
        return
    hash_attr, sort_attr = attrs
    try:
        resp = t.query(
            IndexName=index_name,
            KeyConditionExpression=Key(hash_attr).eq("pending"),
            Limit=limit,
            ScanIndexForward=True,
        )
        fields = ["osf_id", "provider_id", sort_attr, hash_attr]
        for it in resp.get("Items", []):
            print(_pretty({k: it.get(k) for k in fields if k in it}))
    except ClientError as e:
        print(f"[WARN] query {index_name} failed (maybe index missing?): {e}")


def main():
    ap = argparse.ArgumentParser(description="Dump a few rows from local DynamoDB tables")
    ap.add_argument("--table", default=None, help="Specific table to scan (defaults: all known)")
    ap.add_argument("--limit", type=int, default=5, help="Items per scan/query")
    ap.add_argument("--queues", action="store_true", help="Also query queue GSIs if present")
    ap.add_argument("--raw-key", default=None, help="Filter preprints where raw.<path>.<key> exists")
    ap.add_argument("--raw-path", default="attributes", help="Dot path under raw (default: attributes)")
    ap.add_argument("--raw-av", action="store_true", help="Use AttributeValue JSON mode (DynamoDB client)")
    ap.add_argument("--out", default=None, help="Write results to .json or .ndjson file")
    ap.add_argument("--page-limit", type=int, default=None, help="Per-scan page size (default adapts to limit)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep between scan pages (seconds)")
    args = ap.parse_args()

    ddb = get_dynamo_resource()

    preprints_table = _env_name("DDB_TABLE_PREPRINTS", "preprints")

    if args.raw_key:
        if args.table and args.table != preprints_table:
            print(f"[WARN] --raw-key only applies to {preprints_table}; ignoring --table {args.table}")
        writer = _open_writer(args.out) if args.out else None
        try:
            if args.raw_av:
                scan_preprints_with_raw_key_av(
                    args.raw_key,
                    args.limit,
                    raw_path=args.raw_path,
                    writer=writer,
                    progress_every=100 if args.out else 0,
                    page_limit=args.page_limit,
                    sleep_s=args.sleep,
                )
            else:
                scan_preprints_with_raw_key(
                    ddb,
                    args.raw_key,
                    args.limit,
                    raw_path=args.raw_path,
                    writer=writer,
                    progress_every=100 if args.out else 0,
                    page_limit=args.page_limit,
                    sleep_s=args.sleep,
                )
        finally:
            if writer:
                writer.close()
    elif args.table:
        scan_some(ddb, args.table, args.limit)
    else:
        # Known tables (env-overridable names)
        tables: List[str] = [
            preprints_table,
            _env_name("DDB_TABLE_REFERENCES", "preprint_references"),
            _env_name("DDB_TABLE_TEI", "preprint_tei"),
            _env_name("DDB_TABLE_SYNCSTATE", "sync_state"),
            _env_name("DDB_TABLE_API_CACHE", "api_cache"),
        ]

        existing = set(list_tables(ddb))
        print("Tables:", sorted(existing))
        for tn in tables:
            if tn in existing:
                scan_some(ddb, tn, args.limit)
            else:
                print(f"\n=== Table: {tn} (not found) ===")

    if args.queues:
        query_queue(ddb, "by_queue_pdf", args.limit)
        query_queue(ddb, "by_queue_grobid", args.limit)
        query_queue(ddb, "by_queue_extract", args.limit)


def _write_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    # Deprecated: kept for backward compatibility if other modules import it.
    writer = _open_writer(path)
    try:
        for row in rows:
            writer.write_row(row)
    finally:
        writer.close()


class _JsonArrayWriter:
    def __init__(self, f):
        self.f = f
        self.first = True
        self.count = 0
        self.f.write("[")

    def write_row(self, row: Dict[str, Any]) -> None:
        if not self.first:
            self.f.write(",\n")
        else:
            self.first = False
        self.f.write(json.dumps(row, ensure_ascii=False, default=str))
        self.count += 1

    def close(self) -> None:
        self.f.write("]\n")
        self.f.flush()
        print(f"\nWrote {self.count} rows to {self.f.name}")


class _NdjsonWriter:
    def __init__(self, f):
        self.f = f
        self.count = 0

    def write_row(self, row: Dict[str, Any]) -> None:
        self.f.write(json.dumps(row, ensure_ascii=False, default=str))
        self.f.write("\n")
        self.count += 1

    def close(self) -> None:
        self.f.flush()
        print(f"\nWrote {self.count} rows to {self.f.name}")


class _CsvWriter:
    def __init__(self, f):
        self.f = f
        self.count = 0
        self.writer = None

    def write_row(self, row: Dict[str, Any]) -> None:
        if self.writer is None:
            # Preserve insertion order from dict for headers
            self.writer = csv.DictWriter(self.f, fieldnames=list(row.keys()))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.count += 1

    def close(self) -> None:
        self.f.flush()
        print(f"\nWrote {self.count} rows to {self.f.name}")


def _open_writer(path: Optional[str]):
    if not path:
        return None
    if path.lower().endswith(".json"):
        f = open(path, "w", encoding="utf-8")
        return _JsonArrayWriter(f)
    if path.lower().endswith(".csv"):
        f = open(path, "w", encoding="utf-8", newline="")
        return _CsvWriter(f)
    f = open(path, "w", encoding="utf-8")
    return _NdjsonWriter(f)


if __name__ == "__main__":
    main()
