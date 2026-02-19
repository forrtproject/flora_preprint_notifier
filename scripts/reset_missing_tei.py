"""One-off script: find preprints stuck in queue_extract=pending whose TEI
never made it to S3, and reset them back to queue_grobid=pending so the
next GROBID run re-processes and uploads them.

Usage:
    python scripts/reset_missing_tei.py [--dry-run]

Loads AWS credentials from local.env but always targets prod tables.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

import boto3
from dotenv import dotenv_values

# ---------------------------------------------------------------------------
# Load credentials from local.env, override table names to prod
# ---------------------------------------------------------------------------
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
env = dotenv_values(ENV_FILE)

REGION = env.get("AWS_REGION", "eu-central-1")
TABLE_NAME = "prod_preprints"
BUCKET = env.get("TEI_S3_BUCKET", "flora-preprint-tei-cache")

session = boto3.Session(
    aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
    region_name=REGION,
)
ddb = session.resource("dynamodb")
s3 = session.client("s3", region_name=REGION)
table = ddb.Table(TABLE_NAME)


def tei_exists_in_s3(provider_id: str, osf_id: str) -> bool:
    key = f"tei/{provider_id}/{osf_id}/tei.xml"
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def fetch_pending_extract_items():
    """Query the by_queue_extract GSI for all queue_extract=pending items."""
    items = []
    kwargs = {
        "IndexName": "by_queue_extract",
        "KeyConditionExpression": "queue_extract = :q",
        "FilterExpression": "tei_generated = :true",
        "ExpressionAttributeValues": {":q": "pending", ":true": True},
        "ProjectionExpression": "osf_id, provider_id, tei_generated_at",
    }
    while True:
        resp = table.query(**kwargs)
        items.extend(resp.get("Items", []))
        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break
        kwargs["ExclusiveStartKey"] = last_key
    return items


def reset_to_grobid(osf_id: str):
    now = dt.datetime.utcnow().isoformat()
    table.update_item(
        Key={"osf_id": osf_id},
        UpdateExpression=(
            "SET queue_grobid = :pending, tei_generated = :false, updated_at = :t "
            "REMOVE queue_extract, tei_path, tei_generated_at"
        ),
        ExpressionAttributeValues={
            ":pending": "pending",
            ":false": False,
            ":t": now,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Reset preprints with missing S3 TEI back to GROBID queue")
    parser.add_argument("--dry-run", action="store_true", help="Only report, don't modify")
    args = parser.parse_args()

    print(f"Table:  {TABLE_NAME}")
    print(f"Bucket: {BUCKET}")
    print(f"Region: {REGION}")
    print()

    items = fetch_pending_extract_items()
    print(f"Found {len(items)} preprints with queue_extract=pending & tei_generated=True")

    missing = []
    for i, item in enumerate(items):
        osf_id = item["osf_id"]
        provider_id = item.get("provider_id", "")
        if not tei_exists_in_s3(provider_id, osf_id):
            missing.append(item)
        if (i + 1) % 50 == 0:
            print(f"  checked {i + 1}/{len(items)}, {len(missing)} missing so far")

    print(f"\n{len(missing)} preprints have TEI missing from S3")

    if not missing:
        print("Nothing to do.")
        return

    if args.dry_run:
        print("\n[DRY RUN] Would reset these preprints to queue_grobid=pending:")
        for item in missing:
            print(f"  {item['osf_id']}  (provider: {item.get('provider_id', '?')}, tei_at: {item.get('tei_generated_at', '?')})")
        return

    print("\nResetting to queue_grobid=pending ...")
    for i, item in enumerate(missing):
        reset_to_grobid(item["osf_id"])
        if (i + 1) % 10 == 0:
            print(f"  reset {i + 1}/{len(missing)}")

    print(f"Done. Reset {len(missing)} preprints back to GROBID queue.")


if __name__ == "__main__":
    main()
