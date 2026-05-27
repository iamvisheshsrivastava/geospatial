from __future__ import annotations

from pathlib import Path

import boto3


def upload_file_to_s3(local_path: Path, bucket: str, key: str, region_name: str = "us-east-1") -> str:
    client = boto3.client("s3", region_name=region_name)
    client.upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"


def download_file_from_s3(bucket: str, key: str, local_path: Path, region_name: str = "us-east-1") -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client = boto3.client("s3", region_name=region_name)
    client.download_file(bucket, key, str(local_path))
    return local_path
