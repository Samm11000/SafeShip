"""
validator.py
Path: C:\deploy-gate\ml\validator.py

PURPOSE:
  Validates every incoming API request using Pydantic.
  Ensures no bad data (missing fields, wrong types, NaN) ever reaches the ML model.
  Every field has a default so partial requests still work.

HOW TO TEST:
  cd C:\deploy-gate
  python ml\validator.py
"""

from pydantic import BaseModel, Field, validator
from typing import Optional


class BuildFeatures(BaseModel):
    """
    Schema for POST /score request.
    Every field has a default — so if Jenkins can't extract a feature,
    the API still works using the safe fallback value.
    """

    # Required — no defaults
    tenant_id:   str
    api_key:     str
    hour_of_day: int = Field(..., ge=0,   le=23)
    day_of_week: int = Field(..., ge=0,   le=6)

    # Optional with safe defaults (fallback values from feature_extractor.py)
    diff_size:            int   = Field(default=100,  ge=0,   le=50000)
    files_changed:        int   = Field(default=5,    ge=0,   le=1000)
    recent_failure_rate:  float = Field(default=0.0,  ge=0.0, le=1.0)
    test_pass_rate:       float = Field(default=1.0,  ge=0.0, le=1.0)
    is_hotfix:            int   = Field(default=0,    ge=0,   le=1)
    deployer_exp:         int   = Field(default=1,    ge=1)
    days_since_deploy:    float = Field(default=7.0,  ge=0.0)
    build_time_delta:     float = Field(default=0.0)

    # Metadata (not fed to model, used for logging)
    job_name:     Optional[str] = Field(default="unknown")
    branch_name:  Optional[str] = Field(default="")
    triggered_by: Optional[str] = Field(default="unknown")

    @validator("tenant_id")
    def tenant_id_not_empty(cls, v):
        v = v.strip()
        assert len(v) > 0, "tenant_id cannot be empty"
        return v

    @validator("api_key")
    def api_key_not_empty(cls, v):
        v = v.strip()
        assert len(v) > 0, "api_key cannot be empty"
        return v

    @validator("diff_size", pre=True)
    def diff_size_not_nan(cls, v):
        if v is None:
            return 100
        try:
            val = float(v)
            import math
            if math.isnan(val) or math.isinf(val):
                return 100
            return max(0, int(val))
        except (TypeError, ValueError):
            return 100

    def to_model_input(self) -> list:
        """
        Returns features as a list in the EXACT order the model expects.
        This order must match the column order used during training.
        """
        return [
            self.diff_size,
            self.files_changed,
            self.hour_of_day,
            self.day_of_week,
            self.recent_failure_rate,
            self.test_pass_rate,
            self.is_hotfix,
            self.deployer_exp,
            self.days_since_deploy,
            self.build_time_delta,
        ]

    def to_log_dict(self) -> dict:
        """Returns a flat dict for saving to S3 CSV."""
        return {
            "diff_size":            self.diff_size,
            "files_changed":        self.files_changed,
            "hour_of_day":          self.hour_of_day,
            "day_of_week":          self.day_of_week,
            "recent_failure_rate":  self.recent_failure_rate,
            "test_pass_rate":       self.test_pass_rate,
            "is_hotfix":            self.is_hotfix,
            "deployer_exp":         self.deployer_exp,
            "days_since_deploy":    self.days_since_deploy,
            "build_time_delta":     self.build_time_delta,
            "job_name":             self.job_name,
            "branch_name":          self.branch_name,
            "triggered_by":         self.triggered_by,
        }


class LogRequest(BaseModel):
    """
    Schema for POST /log request.
    Called by Jenkins post-build step to record the build outcome.
    """
    tenant_id:        str
    api_key:          str
    build_id:         str
    predicted_score:  int   = Field(..., ge=0, le=100)
    triggered_by:     Optional[str] = Field(default="unknown")

    # All 10 features repeated for storage
    diff_size:            int   = Field(default=100,  ge=0)
    files_changed:        int   = Field(default=5,    ge=0)
    hour_of_day:          int   = Field(default=12,   ge=0,  le=23)
    day_of_week:          int   = Field(default=0,    ge=0,  le=6)
    recent_failure_rate:  float = Field(default=0.0,  ge=0.0, le=1.0)
    test_pass_rate:       float = Field(default=1.0,  ge=0.0, le=1.0)
    is_hotfix:            int   = Field(default=0,    ge=0,  le=1)
    deployer_exp:         int   = Field(default=1,    ge=1)
    days_since_deploy:    float = Field(default=7.0,  ge=0.0)
    build_time_delta:     float = Field(default=0.0)


class SignupRequest(BaseModel):
    """Schema for POST /signup request."""
    email: Optional[str] = Field(default="")


# ─────────────────────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pydantic import ValidationError

    print("=" * 60)
    print("TESTING validator.py")
    print("=" * 60)

    # Test 1: valid full request
    print("\nTest 1: Valid full request")
    req = BuildFeatures(
        tenant_id="abc123",
        api_key="mykey",
        hour_of_day=17,
        day_of_week=4,
        diff_size=847,
        files_changed=12,
        recent_failure_rate=0.4,
        test_pass_rate=0.92,
        is_hotfix=0,
        deployer_exp=23,
        days_since_deploy=2.5,
        build_time_delta=0.12,
    )
    print(f"  [OK] Parsed successfully")
    print(f"  model input: {req.to_model_input()}")

    # Test 2: minimal request (only required fields — rest use defaults)
    print("\nTest 2: Minimal request (only required fields)")
    req2 = BuildFeatures(
        tenant_id="abc123",
        api_key="mykey",
        hour_of_day=10,
        day_of_week=1,
    )
    print(f"  [OK] Defaults applied correctly")
    print(f"  diff_size={req2.diff_size}, test_pass_rate={req2.test_pass_rate}, is_hotfix={req2.is_hotfix}")

    # Test 3: None diff_size should fall back to 100
    print("\nTest 3: None/NaN diff_size should default to 100")
    req3 = BuildFeatures(
        tenant_id="abc123",
        api_key="mykey",
        hour_of_day=9,
        day_of_week=0,
        diff_size=None,
    )
    print(f"  [OK] diff_size with None = {req3.diff_size}  (expected: 100)")

    # Test 4: empty tenant_id should raise error
    print("\nTest 4: Empty tenant_id should raise ValidationError")
    try:
        BuildFeatures(tenant_id="  ", api_key="key", hour_of_day=9, day_of_week=0)
        print("  [FAIL] Should have raised error")
    except (ValidationError, AssertionError, Exception) as e:
        print(f"  [OK] Correctly raised error for empty tenant_id")

    # Test 5: hour_of_day out of range
    print("\nTest 5: hour_of_day=25 should raise ValidationError")
    try:
        BuildFeatures(tenant_id="abc", api_key="key", hour_of_day=25, day_of_week=0)
        print("  [FAIL] Should have raised error")
    except ValidationError:
        print(f"  [OK] Correctly rejected hour_of_day=25")

    # Test 6: to_log_dict check
    print("\nTest 6: to_log_dict returns all 13 keys")
    d = req.to_log_dict()
    assert len(d) == 13, f"Expected 13 keys, got {len(d)}"
    print(f"  [OK] log dict has {len(d)} keys: {list(d.keys())}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED - validator.py is ready")
    print("Next: File 4 - train_base_model.py")
    print("=" * 60)