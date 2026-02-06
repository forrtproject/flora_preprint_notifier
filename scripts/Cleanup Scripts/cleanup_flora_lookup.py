import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Ensure repo root on path when run directly (non-docker)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from osf_sync.dynamo.client import get_dynamo_resource

def is_empty_payload(p):
    if p is None:
        return True
    if isinstance(p, dict) and len(p) == 0:
        return True
    if isinstance(p, str) and p.strip().lower() in {"", "null", "none"}:
        return True
    return False


def contains_none(obj):
    """
    Detect plain None nested anywhere in the structure.
    """
    if obj is None:
        return True
    if isinstance(obj, dict):
        return any(contains_none(v) for v in obj.values())
    if isinstance(obj, list):
        return any(contains_none(v) for v in obj)
    return False


def has_null_marker(obj):
    """
    Detect Dynamo-style {"NULL": true} markers anywhere in the payload.
    """
    # If payload was persisted as a JSON string, try to decode first
    if isinstance(obj, str):
        try:
            import json
            obj = json.loads(obj)
        except Exception:
            # Fallback: look for literal NULL marker substring
            if '"NULL": true' in obj or "'NULL': True" in obj:
                return True
            return False
    # Dynamo JSON shape
    if isinstance(obj, dict):
        if obj.get("NULL") is True:
            return True
        # boto3 AttributeValue style: {"M": {...}} or {"L": [...]} or {"S": "..."}
        if "M" in obj:
            return has_null_marker(obj.get("M"))
        if "L" in obj:
            return has_null_marker(obj.get("L"))
        if "S" in obj:
            return has_null_marker(obj.get("S"))
        return any(has_null_marker(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_null_marker(v) for v in obj)
    return False

ddb = get_dynamo_resource()
table = ddb.Table("preprint_references")

FLORA_FIELDS = [
    "flora_lookup_status",
    "flora_lookup_payload",
    "flora_lookup_original_doi",
    "flora_checked_at",
    "flora_original_cited",
    "flora_ref_pairs",
    "flora_ref_pairs_count",
    "flora_refs",
    "flora_refs_count",
    "flora_matching_replication_dois",
    "flora_lookup_has_output",
    "flora_doi_r_set",
    "flora_apa_ref_o_set",
    "flora_apa_ref_r_set",
]

filter_parts = [f"attribute_exists({f})" for f in FLORA_FIELDS]
scan_kwargs = {
    "FilterExpression": " OR ".join(filter_parts),
    "ProjectionExpression": "osf_id, ref_id, " + ", ".join(FLORA_FIELDS),
}

items = []
resp = table.scan(**scan_kwargs)
items.extend(resp.get("Items", []))
while "LastEvaluatedKey" in resp:
    scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    resp = table.scan(**scan_kwargs)
    items.extend(resp.get("Items", []))

print(f"Found {len(items)} items to inspect")

updated = 0
for it in items:
    osf_id = it["osf_id"]
    ref_id = it["ref_id"]
    removes = list(FLORA_FIELDS)
    sets = []
    eav = {}

    # Nothing to do
    if not removes and not sets:
        continue

    parts = []
    if removes:
        parts.append("REMOVE " + ", ".join(removes))
    if sets:
        parts.append("SET " + ", ".join(sets))
    ue = " ".join(parts)

    kwargs = {
        "Key": {"osf_id": osf_id, "ref_id": ref_id},
        "UpdateExpression": ue,
    }
    if eav:
        kwargs["ExpressionAttributeValues"] = eav

    table.update_item(**kwargs)
    updated += 1
    if updated % 100 == 0:
        print(f"Updated {updated} items so far...")

print(f"Cleanup complete. Updated {updated} items.")
