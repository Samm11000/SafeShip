import os, io, json, time, uuid, boto3, joblib, tempfile
import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from observability import get_logger, set_request_id

log = get_logger("retrain_cron")
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
# Feature contract: one definition, in ml/features.py.
_ml_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml")
if _ml_dir not in sys.path:
    sys.path.insert(0, _ml_dir)
from features import FEATURES   # noqa: E402
def main():
    run_id = uuid.uuid4().hex[:12]
    set_request_id(run_id)   # correlates every line of this run
    log.info("nightly retrain started", extra={"run_id": run_id})

    s3     = boto3.client("s3",         region_name=AWS_REGION)
    dynamo = boto3.resource("dynamodb", region_name=AWS_REGION)
    table  = dynamo.Table(DYNAMO_TABLE)

    tenants   = table.scan().get("Items", [])
    log.info("tenants discovered", extra={"tenant_count": len(tenants)})

    retrained = 0
    skipped   = 0
    failed    = 0

    for tenant in tenants:
        tid = tenant.get("tenant_id", "")
        lc  = int(tenant.get("labelled_count", 0))

        if tid == "demo":
            continue

            log.info("evaluating tenant", extra={"tenant_id": tid, "labelled_count": lc})

        if lc < MIN_BUILDS:
            log.info("skipped: below minimum labelled builds",
                     extra={"tenant_id": tid, "have": lc, "need": MIN_BUILDS})
            skipped += 1
            continue

        try:
            obj     = s3.get_object(Bucket=S3_DATA, Key="tenant_" + tid + "/data.csv")
            df      = pd.read_csv(io.StringIO(obj["Body"].read().decode()))
            df      = df[df["label"].isin([0, 1])].copy()

            if len(df) < MIN_BUILDS:
                log.info("skipped: insufficient rows in S3", extra={"tenant_id": tid})
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

            log.info("candidate model evaluated", extra={
                "tenant_id": tid, "precision": round(prec, 4),
                "auc_roc": round(auc, 4), "rows": len(df), "passed": bool(passed),
            })

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
                log.info("model promoted to production", extra={
                    "tenant_id": tid, "s3_key": model_key,
                    "precision": round(prec, 4), "auc_roc": round(auc, 4),
                })
                retrained += 1
            else:
                log.warning("candidate failed validation; keeping current model", extra={
                    "tenant_id": tid, "precision": round(prec, 4), "auc_roc": round(auc, 4),
                    "min_precision": MIN_PRECISION, "min_auc": MIN_AUC,
                })
                failed += 1

        except Exception as e:
            log.exception("retrain failed for tenant", extra={"tenant_id": tid, "err": str(e)})
            failed += 1

    log.info("nightly retrain finished", extra={
        "run_id": run_id, "retrained": retrained,
        "skipped": skipped, "failed": failed,
    })


if __name__ == "__main__":
    main()