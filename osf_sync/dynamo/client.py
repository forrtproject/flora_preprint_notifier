import os, boto3
from botocore.config import Config

def get_dynamo_resource():
    """
    Switches between local DynamoDB and AWS based on env.
    - For local: set DYNAMO_LOCAL_URL (e.g. http://dynamodb-local:8000)
    - For AWS:   set AWS_REGION and credentials as usual
    """
    local_url = os.getenv("DYNAMO_LOCAL_URL")
    region = os.getenv("AWS_REGION", "eu-central-1")
    cfg = Config(retries={"max_attempts": 10, "mode": "standard"})

    if local_url:
        return boto3.resource(
            "dynamodb",
            region_name=region,
            endpoint_url=local_url,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "dummy"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "dummy"),
            config=cfg,
        )
    return boto3.resource("dynamodb", region_name=region, config=cfg)
