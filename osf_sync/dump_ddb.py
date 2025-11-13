#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
from typing import Optional, List

from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key

# Reuse app's Dynamo client (honors DYNAMO_LOCAL_URL or AWS creds)
from .dynamo.client import get_dynamo_resource


def _env_name(var: str, default: str) -> str:
    return os.getenv(var, default)


def _pretty(x) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2, default=str)


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
    args = ap.parse_args()

    ddb = get_dynamo_resource()

    if args.table:
        scan_some(ddb, args.table, args.limit)
    else:
        # Known tables (env-overridable names)
        tables: List[str] = [
            _env_name("DDB_TABLE_PREPRINTS", "preprints"),
            _env_name("DDB_TABLE_REFERENCES", "preprint_references"),
            _env_name("DDB_TABLE_TEI", "preprint_tei"),
            _env_name("DDB_TABLE_SYNCSTATE", "sync_state"),
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


if __name__ == "__main__":
    main()
