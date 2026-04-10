"""
generate_synthetic.py
Path: C:\\deploy-gate\\ml\\generate_synthetic.py

PURPOSE:
  Generates 3000 rows of realistic synthetic build data for training the base model.
  This is NOT random data - it encodes real DevOps knowledge as rules.

HOW TO RUN:
  cd C:\\deploy-gate
  python ml\\generate_synthetic.py

OUTPUT:
  C:\\deploy-gate\\ml\\data\\synthetic_builds.csv  (3000 rows)
"""

import os
import random
import pandas as pd
import numpy as np

RANDOM_SEED = 42
NUM_ROWS    = 3000
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "synthetic_builds.csv")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def generate_features(n):
    print(f"[generate_synthetic] Generating {n} rows of feature data...")

    diff_size     = np.random.exponential(scale=150, size=n).clip(1, 3000).astype(int)
    files_changed = np.random.exponential(scale=6,   size=n).clip(1, 100).astype(int)

    work_hours  = np.random.randint(9, 18, size=int(n * 0.70))
    offhours    = np.random.randint(0, 24, size=n - int(n * 0.70))
    hour_of_day = np.concatenate([work_hours, offhours])[:n]
    np.random.shuffle(hour_of_day)

    weekdays    = np.random.randint(0, 4, size=int(n * 0.60))
    fridays     = np.full(int(n * 0.25), 4)
    weekends    = np.random.randint(5, 7, size=n - int(n * 0.60) - int(n * 0.25))
    day_of_week = np.concatenate([weekdays, fridays, weekends])[:n]
    np.random.shuffle(day_of_week)

    recent_failure_rate = np.random.beta(a=2, b=5, size=n).clip(0.0, 1.0)
    test_pass_rate      = np.random.beta(a=8, b=2, size=n).clip(0.0, 1.0)
    is_hotfix           = np.random.binomial(n=1, p=0.08, size=n)
    deployer_exp        = np.random.exponential(scale=40, size=n).clip(1, 500).astype(int)
    days_since_deploy   = np.random.exponential(scale=4,  size=n).clip(0.0, 90.0)
    build_time_delta    = np.random.normal(loc=0.0, scale=0.3, size=n).clip(-0.9, 3.0)

    df = pd.DataFrame({
        "diff_size":            diff_size,
        "files_changed":        files_changed,
        "hour_of_day":          hour_of_day,
        "day_of_week":          day_of_week,
        "recent_failure_rate":  recent_failure_rate.round(3),
        "test_pass_rate":       test_pass_rate.round(3),
        "is_hotfix":            is_hotfix,
        "deployer_exp":         deployer_exp,
        "days_since_deploy":    days_since_deploy.round(1),
        "build_time_delta":     build_time_delta.round(3),
    })

    print(f"[generate_synthetic] Features generated. Shape: {df.shape}")
    return df


def assign_labels(df):
    print("[generate_synthetic] Assigning risk labels...")

    n    = len(df)
    risk = np.zeros(n)

    # SIGNAL 1: recent_failure_rate (strongest signal)
    # risk += np.where(df["recent_failure_rate"] > 0.20, 0.30, 0.0)
    risk += np.where(df["recent_failure_rate"] > 0.20, 0.27, 0.0)
    risk += np.where(
        (df["recent_failure_rate"] > 0.10) & (df["recent_failure_rate"] <= 0.20),
        0.15, 0.0
    )

    # SIGNAL 2: is_hotfix
    risk += np.where(df["is_hotfix"] == 1, 0.22, 0.0)

    # SIGNAL 3: test_pass_rate
    risk += np.where(df["test_pass_rate"] < 0.60, 0.20, 0.0)
    risk += np.where(
        (df["test_pass_rate"] >= 0.60) & (df["test_pass_rate"] < 0.75),
        0.10, 0.0
    )

    # SIGNAL 4: diff_size
    risk += np.where(df["diff_size"] > 400, 0.18, 0.0)
    risk += np.where(
        (df["diff_size"] > 150) & (df["diff_size"] <= 400),
        0.08, 0.0
    )

    # SIGNAL 5: hour_of_day
    risk += np.where((df["hour_of_day"] >= 16) & (df["hour_of_day"] <= 19), 0.12, 0.0)
    risk += np.where((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5),  0.10, 0.0)

    # SIGNAL 6: day_of_week
    risk += np.where(df["day_of_week"] == 4, 0.10, 0.0)
    risk += np.where(df["day_of_week"] >= 5, 0.07, 0.0)

    # SIGNAL 7: days_since_deploy
    risk += np.where(df["days_since_deploy"] > 14, 0.08, 0.0)
    risk += np.where(df["days_since_deploy"] > 30, 0.05, 0.0)

    # SIGNAL 8: deployer_exp
    risk += np.where(df["deployer_exp"] <= 5, 0.05, 0.0)

    # SIGNAL 9: files_changed
    risk += np.where(df["files_changed"] > 20, 0.03, 0.0)

    # COMPOUND BOOSTS
    risk += np.where((df["day_of_week"] == 4) & (df["diff_size"] > 200),              0.08, 0.0)
    risk += np.where((df["is_hotfix"] == 1)   & (df["hour_of_day"] >= 16),            0.06, 0.0)
    # risk += np.where((df["recent_failure_rate"] > 0.15) & (df["test_pass_rate"] < 0.80), 0.10, 0.0)
    risk += np.where((df["recent_failure_rate"] > 0.15) & (df["test_pass_rate"] < 0.80), 0.06, 0.0)
    risk += np.where((df["diff_size"] > 300)  & (df["test_pass_rate"] < 0.85),        0.06, 0.0)

    # NOISE
    # noise = np.random.normal(loc=0.0, scale=0.06, size=n)
    noise = np.random.normal(loc=0.0, scale=0.06, size=n)
    risk  = (risk + noise).clip(0.0, 1.0)

    # THRESHOLD: 0.35 gives us ~25% risky builds
    # labels = (risk > 0.35).astype(int)
    # labels = (risk > 0.35).astype(int)
    labels = (risk > 0.44).astype(int)

    df = df.copy()
    df["risk_score_raw"] = risk.round(3)
    df["label"]          = labels
    df["label_source"]   = "synthetic"
    df["sample_weight"]  = 1.0

    risky_count = labels.sum()
    safe_count  = n - risky_count
    print(f"[generate_synthetic] Labels assigned.")
    print(f"  Risky builds (label=1): {risky_count:4d} ({risky_count/n*100:.1f}%)")
    print(f"  Safe  builds (label=0): {safe_count:4d} ({safe_count/n*100:.1f}%)")
    return df


def validate_data(df):
    print("\n[generate_synthetic] Running validation checks...")

    assert len(df) == NUM_ROWS, f"FAIL: Expected {NUM_ROWS} rows, got {len(df)}"
    print(f"  [OK] Row count: {len(df)}")

    nulls = df.isnull().sum().sum()
    assert nulls == 0, f"FAIL: Found {nulls} null values"
    print(f"  [OK] No null values")

    assert df["diff_size"].min() >= 1,                    "FAIL: diff_size < 1"
    assert df["hour_of_day"].between(0, 23).all(),        "FAIL: hour_of_day out of range"
    assert df["day_of_week"].between(0, 6).all(),         "FAIL: day_of_week out of range"
    assert df["recent_failure_rate"].between(0, 1).all(), "FAIL: failure_rate out of range"
    assert df["test_pass_rate"].between(0, 1).all(),      "FAIL: test_pass_rate out of range"
    assert df["is_hotfix"].isin([0, 1]).all(),            "FAIL: is_hotfix bad values"
    print(f"  [OK] All feature ranges valid")

    risky_rate = df["label"].mean()
    assert 0.15 <= risky_rate <= 0.45, \
        f"FAIL: Risky rate {risky_rate:.2f} outside 15-45% range"
    print(f"  [OK] Label distribution: {risky_rate*100:.1f}% risky (target: 15-45%)")

    hotfix_risky_rate = df[df["is_hotfix"] == 1]["label"].mean()
    assert hotfix_risky_rate >= 0.55, \
        f"FAIL: Hotfix risky rate {hotfix_risky_rate:.2f} too low (expected >0.55)"
    print(f"  [OK] Hotfix builds: {hotfix_risky_rate*100:.1f}% risky (expected >55%)")

    high_fail = df[df["recent_failure_rate"] > 0.20]
    if len(high_fail) > 10:
        hf_risky = high_fail["label"].mean()
        assert hf_risky >= 0.55, \
            f"FAIL: High-failure builds {hf_risky:.2f} risky (expected >0.55)"
        print(f"  [OK] High-failure-rate builds: {hf_risky*100:.1f}% risky (expected >55%)")

    fri_aft  = df[(df["day_of_week"] == 4) & (df["hour_of_day"].between(16, 19))]
    tue_morn = df[(df["day_of_week"] == 1) & (df["hour_of_day"].between(9, 12))]
    if len(fri_aft) > 5 and len(tue_morn) > 5:
        fri_rate = fri_aft["label"].mean()
        tue_rate = tue_morn["label"].mean()
        assert fri_rate > tue_rate, \
            f"FAIL: Friday aft ({fri_rate:.2f}) not riskier than Tue morn ({tue_rate:.2f})"
        print(f"  [OK] Friday afternoon {fri_rate*100:.1f}% risky vs Tuesday morning {tue_rate*100:.1f}%")

    print("\n[generate_synthetic] All validation checks PASSED!")


def save_data(df):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_save = df.drop(columns=["risk_score_raw"])
    df_save.to_csv(OUTPUT_PATH, index=False)
    file_size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\n[generate_synthetic] Saved to: {OUTPUT_PATH}")
    print(f"[generate_synthetic] File size: {file_size_kb:.1f} KB")
    print(f"[generate_synthetic] Rows: {len(df_save)}, Columns: {len(df_save.columns)}")
    print(f"\nColumns: {list(df_save.columns)}")


def main():
    print("=" * 60)
    print("GENERATING SYNTHETIC TRAINING DATA")
    print("=" * 60)

    df = generate_features(NUM_ROWS)
    df = assign_labels(df)
    validate_data(df)

    print("\n--- SAMPLE: First 5 rows ---")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df[["diff_size", "hour_of_day", "day_of_week",
              "recent_failure_rate", "is_hotfix",
              "test_pass_rate", "label"]].head())

    print("\n--- FEATURE STATS ---")
    print(df[["diff_size", "recent_failure_rate",
              "test_pass_rate", "days_since_deploy",
              "deployer_exp"]].describe().round(2))

    save_data(df)

    print("\n" + "=" * 60)
    print("SUCCESS - synthetic_builds.csv is ready")
    print("Next: File 3 - validator.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
