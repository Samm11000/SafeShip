import os, io, json, time, boto3, joblib, tempfile
import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, roc_auc_score
from imblearn.over_sampling import SMOTE

AWS_REGION    = "ap-south-1"
S3_MODELS     = "deploy-gate-models"
S3_DATA       = "deploy-gate-data"
DYNAMO_TABLE  = "tenants"
MIN_BUILDS    = 80
MIN_PRECISION = 0.75
MIN_AUC       = 0.70
FEATURES      = [
    "diff_size", "files_changed", "hour_of_day", "day_of_week",
    "recent_failure_rate", "test_pass_rate", "is_hotfix",
    "deployer_exp", "days_since_deploy", "build_time_delta",
]


def main():
    print("\n" + "="*50)
    print("NIGHTLY RETRAIN -- " + datetime.now(timezone.utc).isoformat())
    print("="*50)

    s3     = boto3.client("s3",         region_name=AWS_REGION)
    dynamo = boto3.resource("dynamodb", region_name=AWS_REGION)
    table  = dynamo.Table(DYNAMO_TABLE)

    tenants   = table.scan().get("Items", [])
    print("Found " + str(len(tenants)) + " tenants")

    retrained = 0
    skipped   = 0
    failed    = 0

    for tenant in tenants:
        tid = tenant.get("tenant_id", "")
        lc  = int(tenant.get("labelled_count", 0))

        if tid == "demo":
            continue

        print("\nTenant: " + tid + "  (" + str(lc) + " labelled builds)")

        if lc < MIN_BUILDS:
            print("  Skipped -- need " + str(MIN_BUILDS) + ", have " + str(lc))
            skipped += 1
            continue

        try:
            obj     = s3.get_object(Bucket=S3_DATA, Key="tenant_" + tid + "/data.csv")
            df      = pd.read_csv(io.StringIO(obj["Body"].read().decode()))
            df      = df[df["label"].isin([0, 1])].copy()

            if len(df) < MIN_BUILDS:
                print("  Skipped -- insufficient rows in S3")
                skipped += 1
                continue

            X = df[FEATURES].fillna(0)
            y = df["label"].astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            try:
                k = min(5, int((y_train == 1).sum()) - 1)
                if k >= 1:
                    sm = SMOTE(random_state=42, k_neighbors=k)
                    X_train, y_train = sm.fit_resample(X_train, y_train)
            except Exception:
                pass

            model = RandomForestClassifier(
                n_estimators=100, max_depth=8,
                class_weight="balanced", min_samples_leaf=3,
                random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train)

            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            prec    = precision_score(y_test, y_pred, zero_division=0)
            auc     = roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else 0.5
            passed  = (prec >= MIN_PRECISION and auc >= MIN_AUC
                       and len(df) >= MIN_BUILDS
                       and float(y_test.mean()) >= 0.05)

            print("  precision=" + str(round(prec, 3)) +
                  "  auc=" + str(round(auc, 3)) +
                  "  passed=" + str(passed))

            if passed:
                tmp = tempfile.mktemp(suffix=".pkl")
                joblib.dump(model, tmp)

                cand_key  = "tenant_" + tid + "/candidate.pkl"
                model_key = "tenant_" + tid + "/model.pkl"
                meta_key  = "tenant_" + tid + "/metadata.json"

                s3.upload_file(tmp, S3_MODELS, cand_key)
                s3.copy_object(
                    Bucket=S3_MODELS,
                    CopySource={"Bucket": S3_MODELS, "Key": cand_key},
                    Key=model_key
                )
                s3.delete_object(Bucket=S3_MODELS, Key=cand_key)
                os.remove(tmp)

                meta = {
                    "phase": "tenant", "tenant_id": tid,
                    "trained_at": datetime.now(timezone.utc).isoformat(),
                    "precision": prec, "auc_roc": auc,
                    "dataset_size": len(df)
                }
                s3.put_object(
                    Bucket=S3_MODELS, Key=meta_key,
                    Body=json.dumps(meta).encode(),
                    ContentType="application/json"
                )

                table.update_item(
                    Key={"tenant_id": tid},
                    UpdateExpression="SET model_phase=:p, model_precision=:pr, last_retrain=:t",
                    ExpressionAttributeValues={
                        ":p":  "tenant",
                        ":pr": Decimal(str(round(prec, 4))),
                        ":t":  datetime.now(timezone.utc).isoformat(),
                    }
                )
                print("  SUCCESS -- model swapped")
                retrained += 1
            else:
                print("  FAILED validation -- keeping old model")
                failed += 1

        except Exception as e:
            print("  ERROR: " + str(e))
            failed += 1

    print("\n" + "="*50)
    print("Retrained: " + str(retrained) +
          "  Skipped: " + str(skipped) +
          "  Failed: "  + str(failed))
    print("="*50 + "\n")


if __name__ == "__main__":
    main()