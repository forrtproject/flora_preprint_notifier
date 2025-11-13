import os, logging, boto3
from botocore.config import Config

log = logging.getLogger(__name__)

def _client():
    cfg = Config(retries={"max_attempts": 10, "mode": "standard"})
    endpoint = os.getenv("DYNAMODB_ENDPOINT_URL")
    return boto3.resource(
        "dynamodb",
        region_name=os.getenv("AWS_REGION","eu-north-1"),
        endpoint_url=endpoint,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=cfg,
    )

_dbr = None
def dbr():
    global _dbr
    if _dbr is None:
        _dbr = _client()
    return _dbr

def tbl(name_env, default):
    name = os.getenv(name_env, default)
    return dbr().Table(name)

T_PREPRINTS = lambda: tbl("DDB_TABLE_PREPRINTS", "preprints")
T_REFS      = lambda: tbl("DDB_TABLE_REFERENCES", "preprint_references")
T_SYNC      = lambda: tbl("DDB_TABLE_SYNCSTATE", "sync_state")