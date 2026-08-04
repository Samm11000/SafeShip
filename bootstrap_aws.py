"""
One-time AWS bootstrap for a real SafeShip dry-run.

Creates the S3 buckets + DynamoDB table, uploads the base model, mints a demo
tenant, and writes the credential files the pipelines / act / Jenkins read.

Prereq: AWS credentials configured (env vars or ~/.aws/credentials).
Run:    .venv/bin/python bootstrap_aws.py
"""
import json
import os
import sys

import boto3
from botocore.exceptions import ClientError

REGION       = os.getenv("AWS_REGION", "ap-south-1")
MODELS       = os.getenv("S3_MODELS_BUCKET", "deploy-gate-models")
DATA         = os.getenv("S3_DATA_BUCKET", "deploy-gate-data")
TABLE        = os.getenv("DYNAMO_TABLE", "tenants")

REPO         = os.path.dirname(os.path.abspath(__file__))
DEMO         = os.path.join(os.path.dirname(REPO), "safeship-demo-app")
MODEL_PATH   = os.path.join(REPO, "ml", "data", "base_model.pkl")

sys.path.insert(0, os.path.join(REPO, "app"))
sys.path.insert(0, os.path.join(REPO, "ml"))


def make_bucket(s3, name):
    try:
        s3.create_bucket(Bucket=name, CreateBucketConfiguration={"LocationConstraint": REGION})
        print(f"  created bucket {name}")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print(f"  bucket {name} already exists")
        else:
            raise


def make_table(ddb):
    try:
        ddb.create_table(
            TableName=TABLE,
            KeySchema=[{"AttributeName": "tenant_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "tenant_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        ddb.get_waiter("table_exists").wait(TableName=TABLE)
        print(f"  created table {TABLE}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            print(f"  table {TABLE} already exists")
        else:
            raise


def main():
    print(f"[bootstrap] region={REGION}")
    sts = boto3.client("sts", region_name=REGION)
    ident = sts.get_caller_identity()
    print(f"[bootstrap] AWS account {ident['Account']} as {ident['Arn']}\n")

    s3 = boto3.client("s3", region_name=REGION)
    print("[bootstrap] S3 buckets:")
    make_bucket(s3, MODELS)
    make_bucket(s3, DATA)

    print(f"\n[bootstrap] uploading base model -> s3://{MODELS}/base/model.pkl")
    s3.upload_file(MODEL_PATH, MODELS, "base/model.pkl")

    print(f"\n[bootstrap] DynamoDB:")
    make_table(boto3.client("dynamodb", region_name=REGION))

    import dynamo_client
    creds = dynamo_client.create_tenant(email="dryrun@safeship.test")
    tid, key = creds["tenant_id"], creds["api_key"]

    # Files consumed by the pipelines
    os.makedirs(os.path.join(DEMO, "ci", "jenkins"), exist_ok=True)
    with open(os.path.join(DEMO, "ci", "jenkins", "jenkins.env"), "w") as f:
        f.write(f"SAFESHIP_URL=http://host.docker.internal:5000\n"
                f"SAFESHIP_TENANT_ID={tid}\nSAFESHIP_API_KEY={key}\n")
    with open(os.path.join(DEMO, ".secrets"), "w") as f:   # for `act --secret-file`
        f.write(f"SAFESHIP_TENANT_ID={tid}\nSAFESHIP_API_KEY={key}\n")
    with open(os.path.join(DEMO, ".vars"), "w") as f:      # for `act --var-file`
        f.write("SAFESHIP_URL=http://host.docker.internal:5000\nBREAK=0\n")

    print("\n" + "=" * 60)
    print("  BOOTSTRAP COMPLETE — real AWS is ready")
    print("=" * 60)
    print(f"  tenant_id : {tid}")
    print(f"  api_key   : {key}")
    print(f"  creds written to safeship-demo-app/.secrets, .vars, ci/jenkins/jenkins.env")
    print("=" * 60)


if __name__ == "__main__":
    main()
