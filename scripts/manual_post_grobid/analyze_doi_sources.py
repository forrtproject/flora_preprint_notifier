#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from typing import Dict, Any, Optional

from dotenv import load_dotenv
import os

from osf_sync.db import init_db
from osf_sync.dynamo.client import get_dynamo_resource
from osf_sync.logging_setup import get_logger


log = get_logger("scripts.analyze_doi_sources")


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() in {"true", "1"}
    return bool(value)


def analyze(table_name: str) -> Dict[str, Any]:
    ddb = get_dynamo_resource()
    table = ddb.Table(table_name)

    total = 0
    missing = 0
    by_source: Counter[str] = Counter()
    exclusive_key: Optional[Dict[str, Any]] = None

    projection = "osf_id, ref_id, doi, doi_source, has_doi"

    while True:
        if exclusive_key:
            resp = table.scan(ProjectionExpression=projection, ExclusiveStartKey=exclusive_key)
        else:
            resp = table.scan(ProjectionExpression=projection)
        items = resp.get("Items", [])
        for item in items:
            total += 1
            has_doi = _boolish(item.get("has_doi"))
            doi = (item.get("doi") or "").strip()
            source = item.get("doi_source") or "unspecified"
            if has_doi and doi:
                by_source[source] += 1
            else:
                missing += 1
        exclusive_key = resp.get("LastEvaluatedKey")
        if not exclusive_key:
            break

    return {"total": total, "missing": missing, "by_source": by_source}


def main() -> None:
    parser = argparse.ArgumentParser(description="Show DOI coverage stats per source.")
    parser.add_argument("--table", default=None, help="DynamoDB references table name override.")
    args = parser.parse_args()

    load_dotenv()
    init_db()

    default_table = os.getenv("DDB_TABLE_REFERENCES", "preprint_references")
    table_name = args.table or default_table
    stats = analyze(table_name)

    print("Total references:", stats["total"])
    print("Without DOI:", stats["missing"])
    print("With DOI by source:")
    for source, count in stats["by_source"].most_common():
        print(f"  {source}: {count}")


if __name__ == "__main__":
    main()
