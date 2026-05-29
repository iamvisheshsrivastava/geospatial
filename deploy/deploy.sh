#!/usr/bin/env bash
# Deploy the Satellite Geospatial ML Platform to AWS ECS Fargate.
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Docker running
#   - ECR repository created: satellite-geospatial
#   - ECS cluster created:    satellite-geospatial-cluster
#   - ECS service created:    satellite-geospatial-service
#   - IAM roles: ecsTaskExecutionRole, ecsTaskRole (with S3 + SSM read access)
#   - CloudWatch log group:   /ecs/satellite-geospatial-api
#
# Usage:
#   export AWS_ACCOUNT_ID=123456789012
#   export AWS_REGION=eu-central-1
#   bash deploy/deploy.sh

set -euo pipefail

ACCOUNT_ID="${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}"
REGION="${AWS_REGION:-eu-central-1}"
REPO="satellite-geospatial"
CLUSTER="satellite-geospatial-cluster"
SERVICE="satellite-geospatial-service"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:latest"

echo "==> Authenticating Docker with ECR..."
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin \
    "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> Building Docker image..."
docker build -t "${REPO}:latest" .

echo "==> Tagging and pushing to ECR..."
docker tag "${REPO}:latest" "${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo "==> Registering ECS task definition..."
# Substitute placeholders in task-definition.json
TASK_DEF=$(sed \
  -e "s/ACCOUNT_ID/${ACCOUNT_ID}/g" \
  -e "s/REGION/${REGION}/g" \
  deploy/task-definition.json)

TASK_ARN=$(echo "${TASK_DEF}" \
  | aws ecs register-task-definition \
      --region "${REGION}" \
      --cli-input-json file:///dev/stdin \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['taskDefinition']['taskDefinitionArn'])")

echo "    Registered: ${TASK_ARN}"

echo "==> Updating ECS service to new task revision..."
aws ecs update-service \
  --region "${REGION}" \
  --cluster "${CLUSTER}" \
  --service "${SERVICE}" \
  --task-definition "${TASK_ARN}" \
  --force-new-deployment \
  --output text --query "service.serviceArn"

echo ""
echo "Deployment triggered. Monitor with:"
echo "  aws ecs describe-services --cluster ${CLUSTER} --services ${SERVICE} --region ${REGION}"
echo ""
echo "Or watch in the AWS Console:"
echo "  https://${REGION}.console.aws.amazon.com/ecs/v2/clusters/${CLUSTER}/services/${SERVICE}"
