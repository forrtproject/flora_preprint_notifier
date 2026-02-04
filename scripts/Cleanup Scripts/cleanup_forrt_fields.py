import sys, json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

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

def has_null_marker(obj):
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return '"NULL": true' in obj or "'NULL': True" in obj
    if isinstance(obj, dict):
        if obj.get("NULL") is True:
            return True
        if "M" in obj: return has_null_marker(obj.get("M"))
        if "L" in obj: return has_null_marker(obj.get("L"))
        if "S" in obj: return has_null_marker(obj.get("S"))
        return any(has_null_marker(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_null_marker(v) for v in obj)
    return False

ddb = get_dynamo_resource()
table = ddb.Table("preprint_references")

scan_kwargs = {
    "FilterExpression": (
        "attribute_exists(forrt_refs) OR attribute_exists(forrt_doi_r_set) "
        "OR attribute_exists(forrt_apa_ref_o_set) OR attribute_exists(forrt_apa_ref_r_set) "
        "OR attribute_exists(forrt_lookup_payload) OR attribute_exists(forrt_matching_replication_dois) "
        "OR attribute_exists(forrt_refs_count)"
    ),
    "ProjectionExpression": "osf_id, ref_id, forrt_lookup_payload, forrt_lookup_status, "
                            "forrt_refs, forrt_doi_r_set, forrt_apa_ref_o_set, forrt_apa_ref_r_set, "
                            "forrt_matching_replication_dois, forrt_refs_count",
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
    payload = it.get("forrt_lookup_payload")
    removes = ["forrt_refs", "forrt_doi_r_set", "forrt_apa_ref_o_set", "forrt_apa_ref_r_set",
               "forrt_matching_replication_dois", "forrt_refs_count"]
    sets = []
    eav = {}

    if is_empty_payload(payload) or has_null_marker(payload):
        removes.append("forrt_lookup_payload")
        sets.append("forrt_lookup_status = :s")
        eav[":s"] = False

    # nothing to do
    if not removes and not sets:
        continue

    parts = []
    if removes:
        parts.append("REMOVE " + ", ".join(removes))
    if sets:
        parts.append("SET " + ", ".join(sets))
    ue = " ".join(parts)

    kwargs = {"Key": {"osf_id": osf_id, "ref_id": ref_id}, "UpdateExpression": ue}
    if eav:
        kwargs["ExpressionAttributeValues"] = eav

    table.update_item(**kwargs)
    updated += 1
    if updated % 100 == 0:
        print(f"Updated {updated} items...")

print(f"Cleanup complete. Updated {updated} items.")
