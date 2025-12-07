import os
import urllib.parse

import boto3
from optuna.artifacts import Boto3ArtifactStore


def get_db_conn_str():
    postgres_user = os.getenv("POSTGRES_USER")
    postgres_password = os.getenv("POSTGRES_PASSWORD")
    postgres_host = os.getenv("POSTGRES_HOST")
    postgres_port = os.getenv("POSTGRES_PORT")
    postgres_db = os.getenv("POSTGRES_DB")
    if not all([postgres_user, postgres_password, postgres_host, postgres_port, postgres_db]):
        raise ValueError("One or more PostgreSQL environment variables are not set.")

    safe_user = urllib.parse.quote_plus(postgres_user)
    safe_password = urllib.parse.quote_plus(postgres_password)
    safe_db = urllib.parse.quote_plus(postgres_db)

    return f"postgresql://{safe_user}:{safe_password}@{postgres_host}:{postgres_port}/{safe_db}"

def get_study_name():
    study_name = os.getenv("OPTUNA_STUDY_NAME")
    if not study_name:
        raise ValueError("OPTUNA_STUDY_NAME environment variable is not set.")
    return study_name

def get_artifact_store():
    s3_bucket = os.getenv("OPTUNA_S3_BUCKET")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not s3_bucket:
        raise ValueError("OPTUNA_S3_BUCKET environment variable is not set.")
    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables must be set.")

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name="us-east-1",
    )

    artifact_store = Boto3ArtifactStore(
        client=s3_client,
        bucket_name=s3_bucket,
    )

    return artifact_store
