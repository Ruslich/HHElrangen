import io

import boto3
import pandas as pd


def read_csv_from_s3(bucket, key):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))


# Lightweight helpers for the MVP
_s3 = boto3.client("s3")


def list_keys(bucket: str, prefix: str = "") -> list[str]:
    keys = []
    paginator = _s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            if not item["Key"].endswith("/"):
                keys.append(item["Key"])
    return keys


def read_head_from_s3(bucket: str, key: str, n: int = 20) -> pd.DataFrame:
    obj = _s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"], nrows=n)