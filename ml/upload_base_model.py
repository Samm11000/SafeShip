"""
Upload the locally-trained base model + metadata to S3.

Run AFTER configuring AWS credentials (env vars, ~/.aws/credentials, or a profile).
The model the API serves lives in S3, so this is what actually ships the
sklearn-version-matched retrain to production.

    python3 ml/upload_base_model.py            # uses default creds/region
    AWS_PROFILE=safeship python3 ml/upload_base_model.py
"""
import os
import sys
import boto3

REGION       = os.getenv("AWS_REGION", "ap-south-1")
BUCKET       = os.getenv("S3_MODELS_BUCKET", "deploy-gate-models")
MODEL_KEY    = "base/model.pkl"
META_KEY     = "base/metadata.json"

HERE         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(HERE, "data", "base_model.pkl")
META_PATH    = os.path.join(HERE, "data", "base_metadata.json")


def main():
    for p in (MODEL_PATH, META_PATH):
        if not os.path.exists(p):
            sys.exit(f"[upload] Missing local file: {p}")

    sts = boto3.client("sts", region_name=REGION)
    ident = sts.get_caller_identity()
    print(f"[upload] AWS account {ident['Account']} as {ident['Arn']}")
    print(f"[upload] Target: s3://{BUCKET}/  (region {REGION})")

    s3 = boto3.client("s3", region_name=REGION)
    print(f"[upload] {MODEL_PATH}  ->  s3://{BUCKET}/{MODEL_KEY}")
    s3.upload_file(MODEL_PATH, BUCKET, MODEL_KEY)
    print(f"[upload] {META_PATH}  ->  s3://{BUCKET}/{META_KEY}")
    s3.upload_file(META_PATH, BUCKET, META_KEY)

    # Verify
    s3.head_object(Bucket=BUCKET, Key=MODEL_KEY)
    print("[upload] Verified. Base model refreshed in S3.")


if __name__ == "__main__":
    main()
