#!/usr/bin/env python3
"""One-off script: delete exclusion records for docx_to_pdf_conversion_failed.

After running this, the next OSF sync will re-ingest these preprints and
the PDF stage (with a working LibreOffice) will convert them properly.

Usage:
    # Dry run (default):
    python scripts/rescue_docx_exclusions.py

    # Actually delete:
    python scripts/rescue_docx_exclusions.py --execute
"""

import argparse
import os
import sys

import boto3
from botocore.config import Config

REASON = "docx_to_pdf_conversion_failed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Actually delete (default is dry-run)")
    args = parser.parse_args()

    region = os.getenv("AWS_REGION", "eu-central-1")
    table_name = os.environ.get("DDB_TABLE_EXCLUDED_PREPRINTS", "excluded_preprints")
    cfg = Config(retries={"max_attempts": 10, "mode": "standard"})

    local_url = os.getenv("DYNAMO_LOCAL_URL")
    kwargs = {"region_name": region, "config": cfg}
    if local_url:
        kwargs.update(endpoint_url=local_url,
                      aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "dummy"),
                      aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "dummy"))

    ddb = boto3.resource("dynamodb", **kwargs)
    table = ddb.Table(table_name)

    # Collect all osf_ids with this exclusion reason via the by_reason GSI
    osf_ids = []
    resp = table.query(
        IndexName="by_reason",
        KeyConditionExpression=boto3.dynamodb.conditions.Key("exclusion_reason").eq(REASON),
        ProjectionExpression="osf_id",
    )
    osf_ids.extend(item["osf_id"] for item in resp["Items"])
    while resp.get("LastEvaluatedKey"):
        resp = table.query(
            IndexName="by_reason",
            KeyConditionExpression=boto3.dynamodb.conditions.Key("exclusion_reason").eq(REASON),
            ProjectionExpression="osf_id",
            ExclusiveStartKey=resp["LastEvaluatedKey"],
        )
        osf_ids.extend(item["osf_id"] for item in resp["Items"])

    print(f"Found {len(osf_ids)} exclusion records with reason '{REASON}'")

    if not osf_ids:
        return

    if not args.execute:
        print("Dry run — pass --execute to delete these records")
        return

    with table.batch_writer() as bw:
        for osf_id in osf_ids:
            bw.delete_item(Key={"osf_id": osf_id})

    print(f"Deleted {len(osf_ids)} exclusion records. They will be re-ingested on next sync.")


if __name__ == "__main__":
    main()
