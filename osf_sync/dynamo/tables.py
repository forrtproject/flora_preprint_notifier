from .client import get_dynamo_resource
from botocore.exceptions import ClientError
import time

TABLES = {
  "preprints": {
    "KeySchema":[{"AttributeName":"osf_id","KeyType":"HASH"}],
    "AttributeDefinitions":[
        {"AttributeName":"osf_id","AttributeType":"S"},
        {"AttributeName":"date_published","AttributeType":"S"},
        {"AttributeName":"pdf_downloaded_at","AttributeType":"S"},
        {"AttributeName":"queue_pdf","AttributeType":"S"},
        {"AttributeName":"queue_grobid","AttributeType":"S"},
        {"AttributeName":"queue_extract","AttributeType":"S"}
    ],
    "GlobalSecondaryIndexes":[
        { "IndexName":"by_published",
          "KeySchema":[{"AttributeName":"date_published","KeyType":"HASH"}],
          "Projection":{"ProjectionType":"ALL"},
          "ProvisionedThroughput":{"ReadCapacityUnits":5,"WriteCapacityUnits":5}
        },
        { "IndexName":"by_queue_pdf",
          "KeySchema":[
              {"AttributeName":"queue_pdf","KeyType":"HASH"},
              {"AttributeName":"date_published","KeyType":"RANGE"}
          ],
          "Projection":{"ProjectionType":"ALL"},
          "ProvisionedThroughput":{"ReadCapacityUnits":5,"WriteCapacityUnits":5}
        },
        { "IndexName":"by_queue_grobid",
          "KeySchema":[
              {"AttributeName":"queue_grobid","KeyType":"HASH"},
              {"AttributeName":"pdf_downloaded_at","KeyType":"RANGE"}
          ],
          "Projection":{"ProjectionType":"ALL"},
          "ProvisionedThroughput":{"ReadCapacityUnits":5,"WriteCapacityUnits":5}
        },
        { "IndexName":"by_queue_extract",
          "KeySchema":[
              {"AttributeName":"queue_extract","KeyType":"HASH"},
              {"AttributeName":"date_published","KeyType":"RANGE"}
          ],
          "Projection":{"ProjectionType":"ALL"},
          "ProvisionedThroughput":{"ReadCapacityUnits":5,"WriteCapacityUnits":5}
        }
    ],
    "ProvisionedThroughput":{"ReadCapacityUnits":5,"WriteCapacityUnits":5}
  },
  "preprint_references": {
    "KeySchema":[
        {"AttributeName":"osf_id","KeyType":"HASH"},
        {"AttributeName":"ref_id","KeyType":"RANGE"}
    ],
    "AttributeDefinitions":[
        {"AttributeName":"osf_id","AttributeType":"S"},
        {"AttributeName":"ref_id","AttributeType":"S"},
        {"AttributeName":"doi_source","AttributeType":"S"}
    ],
    "GlobalSecondaryIndexes":[
        { "IndexName":"by_doi_source",
          "KeySchema":[{"AttributeName":"doi_source","KeyType":"HASH"}],
          "Projection":{"ProjectionType":"ALL"},
          "ProvisionedThroughput":{"ReadCapacityUnits":5,"WriteCapacityUnits":5}
        }
    ],
    "ProvisionedThroughput":{"ReadCapacityUnits":5,"WriteCapacityUnits":5}
  },
  "preprint_tei": {
    "KeySchema":[{"AttributeName":"osf_id","KeyType":"HASH"}],
    "AttributeDefinitions":[{"AttributeName":"osf_id","AttributeType":"S"}],
    "ProvisionedThroughput":{"ReadCapacityUnits":5,"WriteCapacityUnits":5}
  },
  "sync_state": {
    "KeySchema":[{"AttributeName":"source_key","KeyType":"HASH"}],
    "AttributeDefinitions":[{"AttributeName":"source_key","AttributeType":"S"}],
    "ProvisionedThroughput":{"ReadCapacityUnits":5,"WriteCapacityUnits":5}
  }
}

def ensure_tables():
    ddb = get_dynamo_resource()
    existing = {t.name for t in ddb.tables.all()}
    for name, spec in TABLES.items():
        if name not in existing:
            params = {"TableName": name, **spec}
            try:
                ddb.create_table(**params).wait_until_exists()
            except ClientError as e:
                if e.response["Error"]["Code"] != "ResourceInUseException":
                    raise
        # Always ensure GSIs exist even for pre-existing tables
        _ensure_gsis(ddb, name, spec)


def _ensure_gsis(ddb, table_name: str, spec: dict) -> None:
    """
    Ensure the table has the GSIs defined in TABLES[table_name].
    Adds any missing GSIs automatically (sequentially). No-op if all present.
    """
    client = ddb.meta.client
    try:
        desc = client.describe_table(TableName=table_name)["Table"]
    except ClientError:
        return

    existing = {g["IndexName"] for g in (desc.get("GlobalSecondaryIndexes") or [])}
    want = [g for g in spec.get("GlobalSecondaryIndexes", [])]
    missing = [g for g in want if g["IndexName"] not in existing]
    if not missing:
        return

    # Attribute definitions currently on the table
    have_attrs = {a["AttributeName"] for a in (desc.get("AttributeDefinitions") or [])}
    spec_attrs = {a["AttributeName"]: a["AttributeType"] for a in spec.get("AttributeDefinitions", [])}

    for gsi in missing:
        # Build AttributeDefinitions for any new key attributes
        needed_attr_names = [k["AttributeName"] for k in gsi.get("KeySchema", [])]
        new_defs = []
        for attr in needed_attr_names:
            if attr not in have_attrs and attr in spec_attrs:
                new_defs.append({"AttributeName": attr, "AttributeType": spec_attrs[attr]})

        params = {
            "TableName": table_name,
            "GlobalSecondaryIndexUpdates": [{"Create": gsi}],
        }
        if new_defs:
            params["AttributeDefinitions"] = new_defs

        try:
            client.update_table(**params)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in {"ResourceInUseException", "ValidationException"}:
                # Likely already creating or already exists due to race; skip to next
                continue
            raise

        # Optionally, wait briefly for index to progress on local dev
        # Avoid long waits on AWS; just short sleep
        time.sleep(0.2)
