"""
inject_test_data.py
Path: C:\deploy-gate\ml\inject_test_data.py

PURPOSE:
  Injects 100 realistic labelled builds into S3 for a tenant.
  Use this to test the full ML pipeline without waiting for real builds.
  Simulates 2 months of realistic build history.

HOW TO RUN:
  cd C:\deploy-gate
  python ml\inject_test_data.py --tenant YOUR_TENANT_ID

  Then trigger retrain manually:
  ssh -i ~/.ssh/deploy-gate-key.pem ubuntu@54.89.160.150
    "docker exec deploy-gate-app python3 /app/app/retrain_cron.py"
"""

import os
import io
import csv
import time
import uuid
import boto3
import argparse
import numpy as np
import pandas as pd
import sys

AWS_REGION = "ap-south-1"
S3_DATA    = "deploy-gate-data"

FEATURE_COLUMNS = [
    "diff_size", "files_changed", "hour_of_day", "day_of_week",
    "recent_failure_rate", "test_pass_rate", "is_hotfix",
    "deployer_exp", "days_since_deploy", "build_time_delta",
]


def generate_realistic_builds(n=100):
    """
    Generates n realistic build rows that simulate a real team.
    Mix of safe and risky builds with realistic patterns.
    """
    np.random.seed(int(time.time()) % 1000)

    now = int(time.time())
    # Spread builds over last 60 days
    timestamps = sorted([
        now - np.random.randint(0, 60 * 86400)
        for _ in range(n)
    ])

    rows = []
    for i, ts in enumerate(timestamps):
        # Realistic feature values
        diff_size           = int(np.random.exponential(150))
        files_changed       = int(np.random.exponential(6))
        hour_of_day         = int(np.random.choice(
            list(range(9, 18)) * 3 + list(range(0, 9)) + list(range(18, 24))
        ))
        day_of_week         = int(np.random.choice(
            [0,1,2,3] * 3 + [4] * 2 + [5, 6]
        ))
        recent_failure_rate = round(float(np.random.beta(2, 6)), 3)
        test_pass_rate      = round(float(np.random.beta(9, 2)), 3)
        is_hotfix           = int(np.random.binomial(1, 0.08))
        deployer_exp        = max(1, i + np.random.randint(1, 10))
        days_since_deploy   = round(float(np.random.exponential(3)), 1)
        build_time_delta    = round(float(np.random.normal(0, 0.2)), 3)

        # Calculate label using same logic as generate_synthetic.py
        risk = 0.0
        risk += 0.28 if recent_failure_rate > 0.20 else (0.14 if recent_failure_rate > 0.10 else 0)
        risk += 0.22 if is_hotfix == 1 else 0
        risk += 0.20 if test_pass_rate < 0.60 else (0.10 if test_pass_rate < 0.75 else 0)
        risk += 0.18 if diff_size > 400 else (0.08 if diff_size > 150 else 0)
        risk += 0.12 if 16 <= hour_of_day <= 19 else (0.10 if hour_of_day >= 22 or hour_of_day <= 5 else 0)
        risk += 0.10 if day_of_week == 4 else (0.07 if day_of_week >= 5 else 0)
        risk += 0.08 if days_since_deploy > 14 else 0
        risk += float(np.random.normal(0, 0.05))
        risk  = max(0, min(1, risk))

        label        = 1 if risk > 0.44 else 0
        label_source = "failure" if label == 1 else "success"
        sample_weight = 1.0

        rows.append({
            "build_id":            f"dg-test-{uuid.uuid4().hex[:8]}",
            "timestamp":           ts,
            "diff_size":           max(1, diff_size),
            "files_changed":       max(1, files_changed),
            "hour_of_day":         max(0, min(23, hour_of_day)),
            "day_of_week":         max(0, min(6, day_of_week)),
            "recent_failure_rate": max(0, min(1, recent_failure_rate)),
            "test_pass_rate":      max(0, min(1, test_pass_rate)),
            "is_hotfix":           is_hotfix,
            "deployer_exp":        int(deployer_exp),
            "days_since_deploy":   max(0, days_since_deploy),
            "build_time_delta":    build_time_delta,
            "predicted_score":     int(risk * 100),
            "label":               label,
            "label_source":        label_source,
            "sample_weight":       sample_weight,
            "triggered_by":        "test-user",
            "job_name":            "test-pipeline",
            "branch_name":         "main",
        })

    df = pd.DataFrame(rows)
    risky = df["label"].sum()
    print(f"  Generated {n} builds: {risky} risky ({risky/n*100:.1f}%), {n-risky} safe")
    return df


def upload_to_s3(tenant_id, df):
    s3  = boto3.client("s3", region_name=AWS_REGION)
    key = f"tenant_{tenant_id}/data.csv"

    # Check if existing data
    try:
        obj      = s3.get_object(Bucket=S3_DATA, Key=key)
        existing = pd.read_csv(io.StringIO(obj["Body"].read().decode()))
        df       = pd.concat([existing, df], ignore_index=True)
        print(f"  Merged with {len(existing)} existing rows -> total {len(df)} rows")
    except Exception:
        print(f"  No existing data — creating new file")

    output = io.StringIO()
    df.to_csv(output, index=False)
    s3.put_object(
        Bucket      = S3_DATA,
        Key         = key,
        Body        = output.getvalue().encode(),
        ContentType = "text/csv",
    )
    print(f"  Uploaded to s3://{S3_DATA}/{key}")
    return len(df)


def update_dynamo(tenant_id, labelled_count):
    dynamo = boto3.resource("dynamodb", region_name=AWS_REGION)
    table  = dynamo.Table("tenants")
    table.update_item(
        Key={"tenant_id": tenant_id},
        UpdateExpression="SET labelled_count=:lc, build_count=:bc",
        ExpressionAttributeValues={
            ":lc": labelled_count,
            ":bc": labelled_count,
        }
    )
    print(f"  DynamoDB updated: labelled_count={labelled_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", required=True, help="Tenant ID to inject data for")
    parser.add_argument("--count",  type=int, default=100, help="Number of builds to inject (default 100)")
    args = parser.parse_args()

    print("="*50)
    print(f"INJECTING TEST BUILD DATA")
    print(f"Tenant : {args.tenant}")
    print(f"Builds : {args.count}")
    print("="*50)

    print("\nStep 1: Generating realistic builds...")
    df = generate_realistic_builds(args.count)

    print("\nStep 2: Uploading to S3...")
    total = upload_to_s3(args.tenant, df)

    print("\nStep 3: Updating DynamoDB...")
    update_dynamo(args.tenant, total)

    print("\n" + "="*50)
    print("DONE — ready to test retrain")
    print("="*50)
    print(f"\nNow run retrain manually:")
    print(f"  ssh -i ~/.ssh/deploy-gate-key.pem ubuntu@54.89.160.150 \\")
    print(f'    "docker exec deploy-gate-app python3 /app/app/retrain_cron.py"')
    print(f"\nThen check your dashboard:")
    print(f"  http://54.89.160.150/dashboard?tenant_id={args.tenant}&api_key=YOUR_KEY")


if __name__ == "__main__":
    main()