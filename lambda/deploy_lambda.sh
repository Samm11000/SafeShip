#!/bin/bash
# deploy_lambda.sh
# Path: C:\deploy-gate\lambda\deploy_lambda.sh
# Upload via S3 to bypass Lambda's 50MB direct upload limit.

set -e

AWS_REGION="ap-south-1"
S3_BUCKET="deploy-gate-models"
RETRAIN_FUNCTION="deploy-gate-retrain"
DRIFT_FUNCTION="deploy-gate-drift"

PROJECT="/mnt/c/Users/swyam/workspace/ProjectsRoot/smart-gate-deploy/deploy-gate"

echo "======================================"
echo "Deploying Lambda functions (via S3)"
echo "======================================"

# ── Get or create Lambda IAM role ──────────────────────────────
echo ""
echo "Step 1: Setting up IAM role..."

ROLE_ARN=$(aws iam get-role --role-name deploy-gate-lambda-role \
            --query 'Role.Arn' --output text 2>/dev/null || echo "")

if [ -z "$ROLE_ARN" ] || [ "$ROLE_ARN" == "None" ]; then
    echo "  Creating Lambda IAM role..."
    aws iam create-role \
        --role-name deploy-gate-lambda-role \
        --assume-role-policy-document '{
            "Version":"2012-10-17",
            "Statement":[{
                "Effect":"Allow",
                "Principal":{"Service":"lambda.amazonaws.com"},
                "Action":"sts:AssumeRole"
            }]
        }' > /dev/null

    aws iam attach-role-policy \
        --role-name deploy-gate-lambda-role \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

    aws iam attach-role-policy \
        --role-name deploy-gate-lambda-role \
        --policy-arn arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess

    aws iam attach-role-policy \
        --role-name deploy-gate-lambda-role \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

    ROLE_ARN=$(aws iam get-role --role-name deploy-gate-lambda-role \
                --query 'Role.Arn' --output text)

    echo "  Waiting 15s for role to propagate..."
    sleep 15
fi
echo "  [OK] Role: $ROLE_ARN"

# ── Package retrain Lambda ──────────────────────────────────────
echo ""
echo "Step 2: Packaging retrain Lambda..."

cd /tmp
rm -rf lambda_retrain lambda_retrain.zip
mkdir lambda_retrain

# Only install what Lambda actually needs (not dev tools)
pip3 install \
    scikit-learn==1.4.2 \
    imbalanced-learn==0.12.3 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scipy==1.13.0 \
    joblib==1.4.2 \
    --target lambda_retrain \
    --quiet \
    --no-deps

# boto3 is pre-installed in Lambda runtime — do NOT include it (saves 30MB)
# numpy is included as dependency of scikit-learn

cp $PROJECT/lambda/retrain/handler.py lambda_retrain/

cd lambda_retrain
zip -r ../lambda_retrain.zip . -q
cd ..

SIZE=$(du -sh lambda_retrain.zip | cut -f1)
echo "  [OK] Packaged. Size: $SIZE"

# ── Upload retrain zip to S3 ────────────────────────────────────
echo ""
echo "Step 3: Uploading retrain package to S3..."
aws s3 cp lambda_retrain.zip s3://$S3_BUCKET/lambda/retrain.zip --region $AWS_REGION
echo "  [OK] Uploaded to s3://$S3_BUCKET/lambda/retrain.zip"

# ── Deploy retrain Lambda ───────────────────────────────────────
echo ""
echo "Step 4: Deploying retrain Lambda..."

ENV_VARS="Variables={S3_MODELS_BUCKET=deploy-gate-models,S3_DATA_BUCKET=deploy-gate-data,DYNAMO_TABLE=tenants}"

if aws lambda get-function --function-name $RETRAIN_FUNCTION --region $AWS_REGION > /dev/null 2>&1; then
    aws lambda update-function-code \
        --function-name $RETRAIN_FUNCTION \
        --s3-bucket $S3_BUCKET \
        --s3-key lambda/retrain.zip \
        --region $AWS_REGION > /dev/null
    echo "  [OK] Retrain Lambda updated"
else
    aws lambda create-function \
        --function-name $RETRAIN_FUNCTION \
        --runtime python3.11 \
        --role $ROLE_ARN \
        --handler handler.lambda_handler \
        --code S3Bucket=$S3_BUCKET,S3Key=lambda/retrain.zip \
        --timeout 300 \
        --memory-size 512 \
        --environment "$ENV_VARS" \
        --region $AWS_REGION > /dev/null
    echo "  [OK] Retrain Lambda created"
fi

# ── Package drift Lambda ────────────────────────────────────────
echo ""
echo "Step 5: Packaging drift Lambda..."

cd /tmp
rm -rf lambda_drift lambda_drift.zip
mkdir lambda_drift

pip3 install \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scipy==1.13.0 \
    --target lambda_drift \
    --quiet \
    --no-deps

cp $PROJECT/lambda/drift/handler.py lambda_drift/

cd lambda_drift
zip -r ../lambda_drift.zip . -q
cd ..

SIZE=$(du -sh lambda_drift.zip | cut -f1)
echo "  [OK] Packaged. Size: $SIZE"

# ── Upload and deploy drift Lambda ─────────────────────────────
echo ""
echo "Step 6: Uploading and deploying drift Lambda..."
aws s3 cp lambda_drift.zip s3://$S3_BUCKET/lambda/drift.zip --region $AWS_REGION

ENV_DRIFT="Variables={S3_DATA_BUCKET=deploy-gate-data,DYNAMO_TABLE=tenants}"

if aws lambda get-function --function-name $DRIFT_FUNCTION --region $AWS_REGION > /dev/null 2>&1; then
    aws lambda update-function-code \
        --function-name $DRIFT_FUNCTION \
        --s3-bucket $S3_BUCKET \
        --s3-key lambda/drift.zip \
        --region $AWS_REGION > /dev/null
    echo "  [OK] Drift Lambda updated"
else
    aws lambda create-function \
        --function-name $DRIFT_FUNCTION \
        --runtime python3.11 \
        --role $ROLE_ARN \
        --handler handler.lambda_handler \
        --code S3Bucket=$S3_BUCKET,S3Key=lambda/drift.zip \
        --timeout 120 \
        --memory-size 256 \
        --environment "$ENV_DRIFT" \
        --region $AWS_REGION > /dev/null
    echo "  [OK] Drift Lambda created"
fi

# ── CloudWatch cron triggers ────────────────────────────────────
echo ""
echo "Step 7: Setting up CloudWatch cron triggers..."

# Retrain: every day 2am UTC
aws events put-rule \
    --name "deploy-gate-nightly-retrain" \
    --schedule-expression "cron(0 2 * * ? *)" \
    --state ENABLED \
    --region $AWS_REGION > /dev/null

RETRAIN_ARN=$(aws lambda get-function \
    --function-name $RETRAIN_FUNCTION \
    --region $AWS_REGION \
    --query 'Configuration.FunctionArn' --output text)

aws events put-targets \
    --rule "deploy-gate-nightly-retrain" \
    --targets "Id=1,Arn=$RETRAIN_ARN" \
    --region $AWS_REGION > /dev/null

aws lambda add-permission \
    --function-name $RETRAIN_FUNCTION \
    --statement-id "allow-cloudwatch-retrain" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --region $AWS_REGION > /dev/null 2>&1 || true

echo "  [OK] Retrain cron: 2am UTC daily"

# Drift: every Sunday 3am UTC
aws events put-rule \
    --name "deploy-gate-weekly-drift" \
    --schedule-expression "cron(0 3 ? * SUN *)" \
    --state ENABLED \
    --region $AWS_REGION > /dev/null

DRIFT_ARN=$(aws lambda get-function \
    --function-name $DRIFT_FUNCTION \
    --region $AWS_REGION \
    --query 'Configuration.FunctionArn' --output text)

aws events put-targets \
    --rule "deploy-gate-weekly-drift" \
    --targets "Id=1,Arn=$DRIFT_ARN" \
    --region $AWS_REGION > /dev/null

aws lambda add-permission \
    --function-name $DRIFT_FUNCTION \
    --statement-id "allow-cloudwatch-drift" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --region $AWS_REGION > /dev/null 2>&1 || true

echo "  [OK] Drift cron: 3am UTC Sundays"

echo ""
echo "======================================"
echo "Lambda deployment complete"
echo "======================================"
echo ""
echo "Test retrain Lambda now:"
echo "  aws lambda invoke --function-name deploy-gate-retrain --region $AWS_REGION /tmp/out.json && cat /tmp/out.json"