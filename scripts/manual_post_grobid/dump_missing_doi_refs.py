#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from osf_sync.db import init_db
from osf_sync.dynamo.client import get_dynamo_resource


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() in {"true", "1"}
    return bool(value)


def dump_missing(table_name: str, output: Optional[str], limit: Optional[int]) -> int:
    ddb = get_dynamo_resource()
    table = ddb.Table(table_name)
    projection = "#oid, ref_id, title, #yr, journal, authors, doi_source, doi, has_doi"
    expr_attr_names = {"#yr": "year", "#oid": "osf_id"}

    dumped = 0
    exclusive_key: Optional[Dict[str, Any]] = None

    if output:
        fh = open(output, "w", encoding="utf-8")
    else:
        fh = None

    try:
        while True:
            if exclusive_key:
                resp = table.scan(
                    ProjectionExpression=projection,
                    ExpressionAttributeNames=expr_attr_names,
                    ExclusiveStartKey=exclusive_key,
                )
            else:
                resp = table.scan(ProjectionExpression=projection, ExpressionAttributeNames=expr_attr_names)
            items = resp.get("Items", [])

            for item in items:
                doi = (item.get("doi") or "").strip()
                has_doi = _boolish(item.get("has_doi"))
                if has_doi and doi:
                    continue

                raw_year = item.get("year")
                if isinstance(raw_year, str):
                    year = raw_year
                else:
                    try:
                        year = int(raw_year) if raw_year is not None else None
                    except Exception:
                        year = str(raw_year) if raw_year is not None else None

                record = {
                    "osf_id": item.get("osf_id"),
                    "ref_id": item.get("ref_id"),
                    "title": item.get("title"),
                    "journal": item.get("journal"),
                    "year": year,
                    "doi_source": item.get("doi_source"),
                }
                text_line = json.dumps(record, ensure_ascii=False)
                if fh:
                    fh.write(text_line + "\n")
                else:
                    print(text_line)
                dumped += 1
                if limit and dumped >= limit:
                    return dumped

            exclusive_key = resp.get("LastEvaluatedKey")
            if not exclusive_key:
                break
        return dumped
    finally:
        if fh:
            fh.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump references missing DOIs after enrichment.")
    parser.add_argument(
        "--table",
        default=None,
        help="References table name (defaults to DDB_TABLE_REFERENCES or preprint_references).",
    )
    parser.add_argument("--output", default=None, help="Write results to file (one JSON per line).")
    parser.add_argument("--limit", type=int, default=None, help="Stop after dumping N entries.")
    args = parser.parse_args()

    load_dotenv()
    init_db()

    table_name = args.table or os.getenv("DDB_TABLE_REFERENCES", "preprint_references")

    count = dump_missing(table_name, args.output, args.limit)
    print(f"Dumped {count} missing-DOI references.")


if __name__ == "__main__":
    main()
