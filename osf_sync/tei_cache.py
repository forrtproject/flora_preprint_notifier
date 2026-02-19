"""Optional S3 cache for TEI XML files.

When TEI_S3_BUCKET is set, TEI files are uploaded after GROBID generation
and downloaded before falling back to expensive GROBID regeneration.
When unset, this module is a no-op.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_BUCKET = os.environ.get("TEI_S3_BUCKET")
_REGION = os.environ.get("AWS_REGION", "eu-central-1")


def _s3_client():
    return boto3.client("s3", region_name=_REGION)


def _key(provider_id: str, osf_id: str) -> str:
    return f"tei/{provider_id}/{osf_id}/tei.xml"


def upload_tei(provider_id: str, osf_id: str, local_path: str) -> bool:
    """Upload a TEI file to S3. Returns True if disabled (no-op), raises on failure."""
    if not _BUCKET:
        return False
    _s3_client().upload_file(local_path, _BUCKET, _key(provider_id, osf_id))
    logger.info("Uploaded TEI to S3 [%s]", osf_id)
    return True


def download_tei(provider_id: str, osf_id: str, local_path: str) -> bool:
    """Download a TEI file from S3. Returns True on success, False if not found or disabled."""
    if not _BUCKET:
        return False
    try:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        _s3_client().download_file(_BUCKET, _key(provider_id, osf_id), local_path)
        logger.info("Downloaded TEI from S3 cache", extra={"osf_id": osf_id})
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "404":
            logger.debug("TEI not in S3 cache", extra={"osf_id": osf_id})
        else:
            logger.warning("Failed to download TEI from S3", extra={"osf_id": osf_id}, exc_info=True)
        return False
